"""Learned identifier detection. Whitespace tokenization, no format regex rules."""
import os,re,unicodedata
from pathlib import Path


def find_crf_model(base_dirs=None):
 """Locate identifier.crfsuite across env override, packaged assets, and repo models/normalization."""
 override=os.environ.get('S2S_CRF_MODEL')
 if override:
  p=Path(override)
  if p.is_file():return p
 here=Path(__file__).resolve()
 pkg_root=here.parents[1]
 roots=[pkg_root,here.parents[2]] if base_dirs is None else [Path(d) for d in base_dirs]
 candidates=[]
 for r in roots:
  candidates.append(r/'assets/identifier.crfsuite')
  candidates.append(r/'models/normalization/identifier.crfsuite')
 for cand in candidates:
  if cand.is_file():return cand
 return pkg_root/'assets/identifier.crfsuite'

def tokens(text):return list(re.finditer(r'\S+',text))
def lexical(s):return ''.join(c.casefold() for c in s if c.isalnum())
def features(ms,i,lang):
 s=ms[i].group();shape=''.join('D' if c.isdecimal() else 'U' if c.isupper() else 'l' if c.isalpha() else c for c in s)
 out={'bias':1.,'lang':lang,'shape':shape,'length':min(len(s),20),'has_digit':any(c.isdecimal() for c in s),'has_alpha':any(c.isalpha() for c in s),'upper':s.isupper()}
 for d in range(-4,5):
  if 0<=i+d<len(ms):
   w=ms[i+d].group();out[f'context{d}']=''.join('D' if c.isdecimal() else c.casefold() for c in w if c.isalnum())
 return out

class IdentifierTagger:
 def __init__(self,path=None):
  import pycrfsuite
  resolved=Path(find_crf_model() if path is None else path)
  if not resolved.is_file():raise FileNotFoundError(f'Missing CRF identifier model: {resolved}')
  self.path=resolved
  self.t=pycrfsuite.Tagger();self.t.open(str(resolved))
 def __call__(self,text,lang):
  ms=tokens(text)
  if not ms:return []
  ys=self.t.tag([features(ms,i,lang) for i in range(len(ms))]);out=[];i=0
  while i<len(ms):
   if ys[i]=='O':i+=1;continue
   a=ms[i].start();j=i+1
   while j<len(ms) and ys[j]=='I':j+=1
   b=ms[j-1].end()
   while a<b and not text[a].isalnum():a+=1
   while b>a and not text[b-1].isalnum():b-=1
   if b>a:out.append(text[a:b])
   i=j
  return out
