import argparse, json, os, pathlib, sys, time, numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'s2s_edge'))
from s2s_edge.neural_textnorm import default_source_mode
from s2s_edge.registry import Registry, resolve_runtime_profile, runtime_code_fingerprint
from s2s_edge.pipeline import Translator
from s2s_edge.audio import load_audio,save_wav,concat_results,segment,effective_max_s,SR

ap=argparse.ArgumentParser(description='Run single-direction E2E translation on an audio file.')
ap.add_argument('pos',nargs='*')
ap.add_argument('--models',default=None,help='Path to models JSON (default: models/models.cpu.json or S2S_MODELS).')
ap.add_argument('--backend',default=None,choices=['cpu','npu'],help='Execution backend (cpu or npu).')
ap.add_argument('--vad-profile',default=None,choices=['legacy','upstream','hybrid'])
ap.add_argument('--ort-policy',default=None,choices=['auto','basic','disable'])
ap.add_argument('--source-mode',default=None,choices=['0','1'])
ap.add_argument('--tts-format',default=None,choices=['hybrid','legacy'])
ap.add_argument('--out-dir',default=None)
args=ap.parse_args()
if len(args.pos)<2:
 ap.error('Expected src tgt [input_wav]')
src,tgt=args.pos[0],args.pos[1]
backend=(args.backend or os.environ.get('S2S_BACKEND') or 'cpu').strip().lower()
models_path=pathlib.Path(args.models) if args.models else (pathlib.Path(os.environ['S2S_MODELS']) if os.environ.get('S2S_MODELS') else (ROOT/'models/models.cpu.json'))
eff_source_mode=args.source_mode if args.source_mode is not None else (os.environ.get('S2S_NEURAL_TN') or default_source_mode(backend))
inputs=ROOT/'results/neural_fix/final/audio'
p=pathlib.Path(args.pos[2]) if len(args.pos)>2 else inputs/('vi_to_en_special_input.wav' if src=='vi' else 'en_to_vi_special_input.wav')
out=pathlib.Path(args.out_dir or os.environ.get('S2S_RESULT_DIR',str(ROOT/'results/e2e')))
(out/'audio').mkdir(parents=True,exist_ok=True)
profile=resolve_runtime_profile(vad_profile=args.vad_profile,ort_policy=args.ort_policy,source_mode=eff_source_mode,tts_format=args.tts_format)
if profile['source_mode']=='1':
 audit=out/f'{src}_to_{tgt}_normalization.jsonl';audit.write_text('')
 os.environ.setdefault('S2S_TN_AUDIT',str(audit))
start=time.perf_counter()
r=Registry(str(models_path),backend,ort_policy=profile['ort_policy'],vad_profile=profile['vad_profile'],source_mode=profile['source_mode'],tts_format=profile['tts_format'])
t=Translator(r,src,tgt,vad_profile=profile['vad_profile'])
raw_asr=[]
class RecordAsr:
 def __init__(self,model):self.model=model
 def __getattr__(self,key):return getattr(self.model,key)
 def __call__(self,wav):
  text=self.model(wav);raw_asr.append(text);return text
t.e._asr[src]=RecordAsr(t.asr)
print('LOADED',src,tgt,time.perf_counter()-start,flush=True)
results=[]
w=load_audio(str(p))
res=segment(w,max_s=effective_max_s(t.asr.max_s,vad_profile=profile['vad_profile']),vad_profile=profile['vad_profile'],return_breaks=True)
spans,brs=res if isinstance(res,tuple) else (res,[[] for _ in res])
for (a,b),br in zip(spans,brs):
 kw={'breaks':br} if br else {}
 z=t.translate_segment(w[a:b],**kw); z['span']=[a/SR,b/SR];z['asr_raw']=raw_asr[-1]
 print('SEGMENT',z['asr'],'=>',z['mt'],z['timing'],flush=True)
 results.append(z)
audio,sr=concat_results(results); path=out/'audio'/f'{src}_to_{tgt}.wav';save_wav(str(path),audio,sr)
data={'src':src,'tgt':tgt,'input':str(p),'output':str(path),'backend':'Qualcomm Hexagon NPU (qnn)' if backend=='npu' else 'ONNX Runtime CPU','execution_backend':backend,'models':str(models_path),'wall_including_load_s':time.perf_counter()-start,'decoder':getattr(t.mt,'active',None),'fallbacks':getattr(t.mt,'failures',[]),'effective_profile':profile,'ort_sessions':r.graph_telemetry(),'code_fingerprint':runtime_code_fingerprint(),'segments':[{k:v for k,v in z.items() if k!='wav'} for z in results]}
data['neural_normalization']=profile['source_mode']=='1'
if data['neural_normalization']:
 from s2s_edge.neural_textnorm import get
 data['normalization_model']=str(get().path)
 data['normalization_device']=get().device
 data['normalization_trace']=get().records
(out/f'{src}_to_{tgt}.json').write_text(json.dumps(data,ensure_ascii=False,indent=2));print('DONE',path,flush=True)
