"""Local mic/file web app -> actual integrated ASR/MT/normalization/TTS.
No browser speech recognition or reference transcripts. One inference worker.
"""
import argparse,gc,io,json,os,sys,time,traceback,uuid,threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse,parse_qs
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'s2s_edge'))
from s2s_edge.neural_textnorm import default_hf_home, default_source_mode
from s2s_edge.registry import resolve_runtime_profile, runtime_code_fingerprint
os.environ.setdefault('S2S_TTS_FORMAT','hybrid')
os.environ.setdefault('S2S_VAD_PROFILE','legacy')
os.environ.setdefault('S2S_ORT_POLICY','auto')
os.environ.setdefault('HF_HOME',default_hf_home())
MAX_BYTES=2*1024*1024

class Service:
 def __init__(self,out,vad_profile=None,ort_policy=None,source_mode=None,tts_format=None,backend=None,models=None,keep_loaded=None):
  self.out=out;out.mkdir(parents=True,exist_ok=True);self.jobs={};self.lock=threading.RLock();self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='s2s')
  self.backend=(backend or os.environ.get('S2S_BACKEND') or 'cpu').strip().lower()
  if self.backend not in ('cpu','npu'):raise ValueError(f'Unsupported backend={self.backend!r}; expected cpu or npu.')
  raw_models=models if models is not None else os.environ.get('S2S_MODELS')
  self.models_path=str(Path(raw_models) if raw_models else (ROOT/'models/models.cpu.json'))
  self.keep_loaded=bool((self.backend=='npu') if keep_loaded is None else keep_loaded)
  self._reg=None;self._translators={}
  eff_source=source_mode if source_mode is not None else (os.environ.get('S2S_NEURAL_TN') or default_source_mode(self.backend))
  self.vad_profile_arg=vad_profile;self.ort_policy_arg=ort_policy;self.source_mode_arg=eff_source;self.tts_format_arg=tts_format
  self.profile=resolve_runtime_profile(vad_profile=vad_profile,ort_policy=ort_policy,source_mode=eff_source,tts_format=tts_format)
  self.loaded_code_fingerprint=runtime_code_fingerprint(extra_files=[str(Path(__file__).resolve())])
  self.code_fingerprint=self.loaded_code_fingerprint
 def close(self):
  if self._reg is not None:
   for k in list(self._reg._cache):self._reg.release(k)
   self._reg=None
  self._translators.clear()
 def update(self,key,**fields):
  with self.lock:self.jobs[key].update(fields)
 def submit(self,data,src,tgt=None):
  import soundfile as sf
  if len(data)>MAX_BYTES:raise ValueError('Bản thu quá lớn. Hãy nói tối đa 30 giây.')
  try:
   info=sf.info(io.BytesIO(data))
  except Exception:raise ValueError('Không đọc được audio. Hãy thu lại hoặc chọn một file âm thanh khác.')
  if info.frames/info.samplerate>30.1 or info.frames/info.samplerate<.25:raise ValueError('Audio cần dài từ 0,25 đến 30 giây.')
  if info.channels>2:raise ValueError('Chỉ nhận audio mono hoặc stereo.')
  tgt_lang=tgt or ('vi' if src=='en' else 'en')
  if tgt_lang not in ('en','vi','ko','zh') or tgt_lang==src:raise ValueError('Ngôn ngữ đích không hợp lệ.')
  with self.lock:
   if sum(j['status'] in ('queued','running') for j in self.jobs.values())>=3:raise ValueError('Máy đang xử lý các bản thu trước. Hãy thử lại sau.')
   key=uuid.uuid4().hex;folder=self.out/key;folder.mkdir();(folder/'input.wav').write_bytes(data)
   self.jobs[key]={'id':key,'status':'queued','stage':'Đang chờ xử lý','src':src,'tgt':tgt_lang,'created':time.time()}
  self.pool.submit(self.run,key);return key
 def run(self,key):
  from s2s_edge.registry import Registry
  from s2s_edge.pipeline import Translator
  from s2s_edge.audio import load_audio,save_wav,segment,concat_results,effective_max_s,SR
  from s2s_edge import neural_textnorm
  import numpy as np
  reg=self._reg;translator=None;saved_hooks=None;folder=self.out/key;start=time.perf_counter()
  with self.lock:src,tgt=self.jobs[key]['src'],self.jobs[key]['tgt']
  try:
   self.update(key,status='running',stage='Đang chuẩn bị mô hình…')
   os.environ['S2S_TN_AUDIT']=str(folder/'normalization.jsonl')
   w=load_audio(str(folder/'input.wav'))
   if not len(w) or not np.isfinite(w).all() or float(np.max(abs(w)))<1e-5:raise ValueError('Bản thu không có tín hiệu. Hãy kiểm tra mic và nói lại.')
   profile=dict(self.profile)
   if reg is None:
    reg=Registry(self.models_path,self.backend,ort_policy=profile['ort_policy'],vad_profile=profile['vad_profile'],source_mode=profile['source_mode'],tts_format=profile['tts_format'])
    if self.keep_loaded:self._reg=reg
   translator=self._translators.get((src,tgt))
   if translator is None:
    translator=Translator(reg,src,tgt,vad_profile=profile['vad_profile'])
    if self.keep_loaded:self._translators[(src,tgt)]=translator
   raw=[];service=self
   class ASR:
    def __init__(self,m):self.m=m
    def __getattr__(self,n):return getattr(self.m,n)
    def __call__(self,audio):
     service.update(key,stage='Đang nhận dạng giọng nói…');text=self.m(audio);raw.append(text);service.update(key,stage='Đang chuẩn hóa văn bản nguồn…');return text
   class MT:
    def __init__(self,m):self.m=m
    def __getattr__(self,n):return getattr(self.m,n)
    def translate_atomic(self,*a):service.update(key,stage='Đang dịch…');return self.m.translate_atomic(*a)
    def __call__(self,*a):service.update(key,stage='Đang dịch…');return self.m(*a)
   class TTS:
    def __init__(self,m):self.m=m
    def __getattr__(self,n):return getattr(self.m,n)
    def synthesize(self,text):service.update(key,stage='Đang chuẩn hóa và tạo giọng nói…');return self.m.synthesize(text)
   saved_hooks=(translator.e._asr.get(src,translator.asr),translator.e._mt,translator.e._tts.get(tgt,translator.tts))
   translator.e._asr[src]=ASR(saved_hooks[0]);translator.e._mt=MT(saved_hooks[1]);translator.e._tts[tgt]=TTS(saved_hooks[2])
   res=segment(w,max_s=effective_max_s(translator.asr.max_s,vad_profile=profile['vad_profile']),vad_profile=profile['vad_profile'],return_breaks=True)
   spans,brs=res if isinstance(res,tuple) else (res,[[] for _ in res]);results=[]
   if not spans:raise ValueError('Chưa nhận ra lời nói. Hãy nói rõ hơn và đưa mic lại gần.')
   neural_active=translator.e.neural_enabled()
   trace_start=len(neural_textnorm.get().records) if neural_active else 0
   for i,((a,b),br) in enumerate(zip(spans,brs)):
    self.update(key,segment=i+1,segments_total=len(spans))
    kw={'breaks':br} if br else {}
    result=translator.translate_segment(w[a:b],**kw);result['asr_raw']=raw[-1];result['span']=[a/SR,b/SR];results.append(result)
   ledger_entries=[]
   audit_totals={'total':0,'preserved':0,'mismatch':0,'unverifiable':0}
   for idx_seg,z in enumerate(results):
    ta=z.get('target_audit') or {}
    cnt=ta.get('counts') or {}
    for k_cnt in audit_totals:audit_totals[k_cnt]+=int(cnt.get(k_cnt,0))
    ledger_entries.append({
     'job_id':key,'segment_index':idx_seg,'span':z.get('span'),'src':src,'tgt':tgt,
     'source_mode':profile['source_mode'],'execution_backend':self.backend,
     'asr_raw':z.get('asr_raw',''),'asr_before_normalization':z.get('asr_before_normalization',z.get('asr_raw','')),
     'prepared':z.get('asr',''),'mt':z.get('mt',''),'tts_text':z.get('tts_text',''),
     'source_guard':z.get('source_guard',{}),'target_audit':ta,
     'effective_profile':profile,'code_fingerprint':self.code_fingerprint,
    })
   (folder/'audit_ledger.jsonl').write_text(''.join(json.dumps(e,ensure_ascii=False)+'\n' for e in ledger_entries),encoding='utf-8')
   audio,sr=concat_results(results)
   if not len(audio):raise ValueError('Không nhận được nội dung để dịch. Hãy nói lại một câu ngắn, rõ hơn.')
   if not np.isfinite(audio).all():raise RuntimeError('Mô hình trả về audio không hợp lệ.')
   save_wav(str(folder/'output.wav'),audio,sr)
   if self.backend=='npu':
    backend_label='Qualcomm Hexagon NPU (qnn) + learned normalization' if neural_active else 'Qualcomm Hexagon NPU (qnn) (edge preparation; neural_tn=0)'
   else:
    backend_label='integrated ONNX CPU + learned normalization on Apple Metal' if neural_active else 'integrated ONNX CPU (rule source preparation; neural_tn=0)'
   result={'id':key,'status':'done','stage':'Hoàn tất','src':src,'tgt':tgt,'input_seconds':len(w)/SR,'output_seconds':len(audio)/sr,'elapsed_seconds':time.perf_counter()-start,'input_url':f'/audio/{key}/input.wav','output_url':f'/audio/{key}/output.wav','detail_url':f'/audio/{key}/result.json','asr':' '.join(z['asr_raw'] for z in results),'translation':' '.join(z['mt'] for z in results),'tts_text':' '.join(z['tts_text'] for z in results),'segments':[{k:v for k,v in z.items() if k!='wav'} for z in results],'normalization':neural_textnorm.get().records[trace_start:] if neural_active else [],'audit_ledger':ledger_entries,'target_audit_summary':audit_totals,'source_mode':profile['source_mode'],'effective_profile':profile,'ort_sessions':reg.graph_telemetry(),'code_fingerprint':self.code_fingerprint,'backend':backend_label,'execution_backend':self.backend,'models':self.models_path,'input_kind':'uploaded speech audio; no transcript provided'}
   (folder/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));self.update(key,**{k:v for k,v in result.items() if k!='id'})
   print('DONE',key,src,tgt,round(result['elapsed_seconds'],2),flush=True)
  except Exception as e:
   traceback.print_exc();self.update(key,status='error',stage='Chưa xử lý được bản thu',error=str(e)[:500])
   with self.lock:(folder/'error.json').write_text(json.dumps(self.jobs[key],ensure_ascii=False,indent=2))
  finally:
   if saved_hooks is not None and translator is not None:
    translator.e._asr[src],translator.e._mt,translator.e._tts[tgt]=saved_hooks
   if not self.keep_loaded and reg is not None:
    for k in list(reg._cache):reg.release(k)
   gc.collect()
 def snapshot(self,key):
  with self.lock:
   j=self.jobs.get(key)
   return dict(j) if j else None

def build_handler(service, port):
 class Handler(BaseHTTPRequestHandler):
  def send(self,status,body,ctype='application/json; charset=utf-8'):
   self.send_response(status);self.send_header('Content-Type',ctype);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('Permissions-Policy','microphone=(self)');self.end_headers()
   try:self.wfile.write(body)
   except (BrokenPipeError,ConnectionResetError):pass
  def json(self,status,value):self.send(status,json.dumps(value,ensure_ascii=False).encode())
  def log_message(self,fmt,*args):
   if self.path!='/api/health' and not self.path.startswith('/api/jobs/'):print(fmt%args,flush=True)
  def do_GET(self):
   path=urlparse(self.path).path
   static={'/':('index.html','text/html; charset=utf-8'),'/app.js':('app.js','text/javascript; charset=utf-8'),'/style.css':('style.css','text/css; charset=utf-8'),'/recorder.js':('recorder.js','text/javascript; charset=utf-8')}
   if path in static:
    name,ctype=static[path];return self.send(200,(ROOT/'web'/name).read_bytes(),ctype)
   if path=='/api/health':
    with service.lock:pending=sum(j['status'] in ('queued','running') for j in service.jobs.values())
    p=service.profile
    return self.json(200,{'ok':True,'pid':os.getpid(),'pending':pending,'max_seconds':30,'backend':service.backend,'models':service.models_path,'keep_loaded':service.keep_loaded,'normalization':p['source_mode']=='1','source_mode':p['source_mode'],'vad_profile':p['vad_profile'],'vad_options':p['vad_options'],'ort_policy':p['ort_policy'],'tts_format':p['tts_format'],'effective_profile':dict(p),'loaded_code_fingerprint':service.loaded_code_fingerprint,'code_fingerprint':service.code_fingerprint})
   if path.startswith('/api/jobs/'):
    j=service.snapshot(path.rsplit('/',1)[-1]);return self.json(200,j) if j else self.json(404,{'error':'Không tìm thấy bản thu.'})
   if path.startswith('/audio/'):
    parts=path.split('/')
    if len(parts)!=4 or parts[3] not in ('input.wav','output.wav','result.json'):return self.json(404,{'error':'Not found'})
    try:uuid.UUID(hex=parts[2])
    except ValueError:return self.json(404,{'error':'Not found'})
    p=service.out/parts[2]/parts[3]
    if not p.is_file():return self.json(404,{'error':'Audio chưa sẵn sàng.'})
    return self.send(200,p.read_bytes(),'audio/wav' if p.suffix=='.wav' else 'application/json; charset=utf-8')
   return self.json(404,{'error':'Not found'})
  def do_POST(self):
   if urlparse(self.path).path!='/api/jobs':return self.json(404,{'error':'Not found'})
   origin=self.headers.get('Origin')
   if origin and origin not in (f'http://localhost:{port}',f'http://127.0.0.1:{port}'):return self.json(403,{'error':'Hãy mở trang trên localhost.'})
   try:
    qs=parse_qs(urlparse(self.path).query)
    src=qs.get('src',[''])[0]
    tgt=qs.get('tgt',[''])[0] or None
    if src not in ('en','vi'):raise ValueError('Hãy chọn chiều Anh → Việt hoặc Việt → Anh.')
    n=int(self.headers.get('Content-Length','0'))
    if n<=0 or n>MAX_BYTES:raise ValueError('Kích thước audio không hợp lệ; tối đa 30 giây.')
    data=self.rfile.read(n)
    if not data.startswith(b'RIFF') or data[8:12]!=b'WAVE':raise ValueError('Hãy gửi audio WAV; trình duyệt sẽ tự chuyển đổi bản thu.')
    key=service.submit(data,src,tgt=tgt);self.json(202,{'id':key})
   except ValueError as e:self.json(400,{'error':str(e)})
   except Exception:
    traceback.print_exc();self.json(500,{'error':'Không gửi được bản thu. Hãy thử lại.'})
 return Handler

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--port',type=int,default=8766);ap.add_argument('--host',default='127.0.0.1');ap.add_argument('--out-dir',default=None);ap.add_argument('--models',default=None);ap.add_argument('--backend',default=None,choices=['cpu','npu']);ap.add_argument('--keep-loaded',action='store_true');ap.add_argument('--vad-profile',default=None,choices=['legacy','upstream','hybrid']);ap.add_argument('--ort-policy',default=None,choices=['auto','basic','disable']);ap.add_argument('--source-mode',default=None,choices=['0','1']);ap.add_argument('--tts-format',default=None,choices=['hybrid','legacy']);args=ap.parse_args()
 service=Service(Path(args.out_dir) if args.out_dir else (ROOT/'results/mic_web'),vad_profile=args.vad_profile,ort_policy=args.ort_policy,source_mode=args.source_mode,tts_format=args.tts_format,backend=args.backend,models=args.models,keep_loaded=True if args.keep_loaded else None)
 Handler=build_handler(service, args.port)
 print(f'OneVoice mic web ({service.backend}): http://localhost:{args.port}',flush=True)
 ThreadingHTTPServer((args.host,args.port),Handler).serve_forever()

if __name__=='__main__':main()
