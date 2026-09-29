"""Read-only browser view of native camera perception and existing safety output."""
import json
import math
import threading
import time
from urllib.parse import urlparse, parse_qs
from .scenario_camera_status import StatusStore


class CameraObservation:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.lock = threading.RLock()
        self.store = StatusStore()
        self.policy = None
        self.policy_expires = float('-inf')

    def _receive(self, kind, message, source_age_s):
        now = self.clock()
        with self.lock:
            if not math.isfinite(source_age_s) or not 0 <= source_age_s < .5:
                setattr(self.store, kind + '_at', float('-inf'))
                setattr(self.store, kind + '_error', kind + '_source_missing_or_expired')
                return
            getattr(self.store, kind)(message, now - source_age_s)

    def rgb(self, message, source_age_s):
        self._receive('rgb', message, source_age_s)

    def depth(self, message, source_age_s):
        self._receive('depth', message, source_age_s)

    def safety(self, message, source_age_s):
        now = self.clock()
        with self.lock:
            age = source_age_s + float(message.observation_age_s)
            if (not math.isfinite(source_age_s) or not math.isfinite(age)
                    or source_age_s < 0 or message.observation_age_s < 0 or not 0 <= age < .5):
                self.policy_expires = float('-inf')
                return
            state = {0: 'CLEAR', 1: 'SLOW', 2: 'HOLD', 3: 'EVADE', 4: 'STALE'}.get(message.state, 'STALE')
            self.policy = {'state': state, 'reason': message.reason, 'source': message.source}
            self.policy_expires = now + .5 - age

    def snapshot(self, nonce=''):
        now = self.clock()
        with self.lock:
            value = self.store.snapshot(now)
            # Both streams must be current for a live camera label.
            remaining = min(value['sensor_valid_for_s'],
                            max(0., .5 - (now - self.store.rgb_at)))
            if not value['camera_ok']:
                value['perception'].update(assessment='UNKNOWN',
                    reason='camera_rgb_or_depth_missing_or_expired', valid_for_s=0.)
                remaining = 0.
            value['perception']['valid_for_s'] = remaining
            value['display_valid_for_s'] = remaining
            value['request_nonce'] = nonce
            if self.policy is not None and now < self.policy_expires:
                value['movement_policy'] = dict(self.policy,
                    valid_for_s=max(0., self.policy_expires - now))
            else:
                value['movement_policy'] = {'state': 'UNKNOWN',
                    'reason': 'safety_decision_missing_or_expired', 'valid_for_s': 0.}
            return value


def handler_with_observation(base_handler, observation, bench):
    class Handler(base_handler):
        def do_GET(self):
            parsed = urlparse(self.path)
            nonce = parse_qs(parsed.query).get('nonce', [''])[0]
            if len(nonce) > 128:
                self.send_error(400)
                return
            if parsed.path in ('/', '/observe', '/observe/'):
                body, mime = VIEW_HTML.encode('utf-8'), 'text/html; charset=utf-8'
            elif parsed.path == '/observe/v1/status':
                body = json.dumps(observation.snapshot(nonce), allow_nan=False).encode()
                mime = 'application/json'
            elif parsed.path == '/monitor/v1/status':
                value = bench.monitor_snapshot(nonce)
                value['camera_observation'] = observation.snapshot(nonce)
                body, mime = json.dumps(value, allow_nan=False).encode(), 'application/json'
            else:
                return super().do_GET()
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass
    return Handler


VIEW_HTML = r'''<!doctype html>
<html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Jetson Camera Observe</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#111820;color:#e6edf3;font:16px system-ui,sans-serif}
header{padding:20px 28px;border-bottom:1px solid #33404d;display:flex;justify-content:space-between;gap:16px}
h1{font-size:22px;margin:0}header span{color:#98adbf;font-size:14px}main{padding:24px;display:grid;grid-template-columns:minmax(0,1.6fr) minmax(320px,1fr);gap:28px;max-width:1600px;margin:auto}
figure{margin:0;position:relative;align-self:start;background:#080d12;aspect-ratio:4/3;overflow:hidden}img{width:100%;height:100%;object-fit:contain}figcaption{position:absolute;bottom:0;padding:10px 14px;background:#111820df;width:100%;font-size:13px;color:#aabccb}
.label{color:#98adbf;font-size:14px;margin:0 0 8px}.state{font-size:42px;font-weight:750;line-height:1.15;color:#aabccb}.state.CLEAR{color:#58d799}.state.BLOCKED{color:#ff7777}.state.CLIMB_REQUIRED{color:#ffc66d}
#meaning{margin:10px 0 8px}#reason{font-size:13px;color:#9aafc1;overflow-wrap:anywhere;min-height:36px}
dl{margin:20px 0;display:grid;grid-template-columns:1fr auto;gap:10px;font-size:15px}dt{color:#98adbf}dd{margin:0;font-variant-numeric:tabular-nums}
.policy{padding-top:20px;border-top:1px solid #33404d}.policy b{display:block;font-size:22px;margin:8px 0}small{color:#98adbf;font-size:13px}#incident{margin-top:24px;padding-top:16px;border-top:1px solid #33404d}#hint{color:#ffc66d;font-size:14px}#video-status{color:#d3e0eb}
@media(max-width:900px){main{grid-template-columns:1fr}header{flex-direction:column}main{padding:16px}.state{font-size:36px}}
</style>
<header><h1>Jetson · Camera Observe</h1><span>2m 감지 · 최대 +2m 상승 · CLEAR 후 3m 전진 · 원고도 복귀</span></header>
<main><figure><img id="preview" alt="D435 실시간 카메라 영상"><figcaption id="video-status">영상 연결 중…</figcaption></figure>
<aside><p class="label">현재 카메라 장애물 판정</p><div id="assessment" class="state UNKNOWN">UNKNOWN</div>
<p id="meaning">카메라 데이터 대기 중</p><div id="reason">waiting_for_camera</div><p id="hint"></p>
<dl><dt>전방 근거리</dt><dd id="near">—</dd><dt>전방 중앙값</dt><dd id="median">—</dd><dt>상부 ROI 거리</dt><dd id="upper">—</dd><dt>전체 깊이 유효율</dt><dd id="quality">—</dd><dt>중앙 / 상부 / 하부 유효율</dt><dd id="roi-quality">—</dd><dt>전방 근접 픽셀 비율</dt><dd id="fraction">—</dd><dt>깊이 수신 경과</dt><dd id="age">—</dd></dl>
<div class="policy"><small>이동 정책 (기존 시나리오 판단)</small><b id="policy">UNKNOWN</b><small id="policy-reason">수신 대기 중</small><p><small>카메라 관측은 이동 허가가 아닙니다. CLIMB_REQUIRED는 재관측 후보이며 상부 비행 경로는 미검증입니다.</small></p></div>
<div id="incident">사건 추론 연결 중…</div></aside></main>
<script>
const el=id=>document.getElementById(id);let deadline=0,policyDeadline=0,videoDeadline=0;
const meanings={CLEAR:'전방 장애물 미감지',BLOCKED:'전방 장애물 감지',CLIMB_REQUIRED:'전방 장애물 감지 · 상부 재관측 후보',UNKNOWN:'판정 불가 / 데이터 없음'};
const metric=(v,unit,digits=2)=>typeof v==='number'&&Number.isFinite(v)?v.toFixed(digits)+unit:'—';
const percent=v=>typeof v==='number'&&Number.isFinite(v)?metric(v*100,'%',1):'—';
function state(name,reason){const valid=Object.hasOwn(meanings,name)?name:'UNKNOWN';el('assessment').textContent=valid;el('assessment').className='state '+valid;el('meaning').textContent=meanings[valid];el('reason').textContent=reason||'—';el('hint').textContent=valid==='CLIMB_REQUIRED'?'승인 고도 내 최대 +2m 상승 · CLEAR 정지 후 3m 전진 (비행 경로 미검증)':'';}
function clearMetrics(){for(const id of ['near','median','upper','quality','roi-quality','fraction','age'])el(id).textContent='—';}
function expire(reason='camera_data_expired'){deadline=0;state('UNKNOWN',reason);clearMetrics();}
async function fetchStatus(path){const nonce=crypto.randomUUID();const start=performance.now();const response=await fetch(path+'?nonce='+nonce,{cache:'no-store',signal:AbortSignal.timeout(1500)});if(!response.ok)throw Error('HTTP '+response.status);const value=await response.json();if(value.request_nonce!==nonce)throw Error('response_nonce_mismatch');return {value,start};}
async function pollCamera(){try{const {value:v,start}=await fetchStatus('/observe/v1/status');const p=v.perception||{},s=v.sensor||{};deadline=start+Math.min(.5,Math.max(0,v.display_valid_for_s||0))*1000;if(v.camera_ok!==true||performance.now()>=deadline){expire(p.reason||'camera_data_expired');}else{state(p.assessment,p.reason);el('near').textContent=metric(s.front_near_m,' m');el('median').textContent=metric(s.front_median_m,' m');el('upper').textContent=metric(s.upper_roi_near_m,' m');el('quality').textContent=percent(s.valid_fraction);el('roi-quality').textContent=[s.center_valid_fraction,s.upper_valid_fraction,s.lower_valid_fraction].map(percent).join(' / ');el('fraction').textContent=percent(s.front_obstacle_fraction);el('age').textContent=metric(v.receipt_age_s.depth*1000,' ms',0);}const policy=v.movement_policy||{};policyDeadline=start+Math.min(.5,Math.max(0,policy.valid_for_s||0))*1000;el('policy').textContent=performance.now()<policyDeadline?policy.state:'UNKNOWN';el('policy-reason').textContent=policy.reason||'safety_decision_missing_or_expired';}catch(e){expire('camera_connection_unavailable');policyDeadline=0;el('policy').textContent='UNKNOWN';el('policy-reason').textContent='safety_connection_unavailable';}finally{setTimeout(pollCamera,100);}}
async function pollIncident(){try{const {value:v,start}=await fetchStatus('/monitor/v1/status');const current=v.status==='RUNNING'&&performance.now()<start+Math.max(0,v.display_valid_for_s||0)*1000;el('incident').textContent=current?`사건 추론: ${v.status} · 사람 ${(v.people||[]).length}명 · 투기 후보 ${(v.littering_predictions||[]).length} · 현재 확정 ${(v.confirmed_current_frame||[]).length}건`:'사건 추론: '+v.status+' · 현재 결과 대기';}catch(e){el('incident').textContent='사건 추론: 연결 끊김';}finally{setTimeout(pollIncident,500);}}
let imageUrl;
async function getImage(path){
 const start=performance.now();
 const response=await fetch(path+'?t='+Date.now(),{cache:'no-store',signal:AbortSignal.timeout(1500)});
 if(!response.ok)throw Error('preview_unavailable');
 const header=response.headers.get('X-Frame-Age-Ms');const age=header===null?NaN:Number(header);
 const expires=start+1000-age;
 if(!Number.isFinite(age)||age<0)throw Error('preview_age_invalid');
 const url=URL.createObjectURL(await response.blob());
 try{const decoded=new Image();decoded.src=url;await decoded.decode();if(performance.now()>=expires)throw Error('preview_expired');return {url,expires};}
 catch(e){URL.revokeObjectURL(url);throw e;}
}
let annotatedDeadline=0;
async function pollVideo(annotated=true){
 try{
  const value=await getImage(annotated?'/monitor/v1/preview.jpg':'/preview.jpg');
  if(!annotated&&performance.now()<annotatedDeadline){URL.revokeObjectURL(value.url);return;}
  if(annotated)annotatedDeadline=value.expires;
  const old=imageUrl;imageUrl=value.url;videoDeadline=value.expires;
  el('preview').src=imageUrl;el('preview').style.opacity='1';
  el('video-status').textContent=annotated?'영상 · 사람/투기 주석 (최대 1초 지연)':'원본 영상 · 주석 대기';
  if(old)URL.revokeObjectURL(old);
 }catch(e){if(performance.now()>=videoDeadline){el('preview').style.opacity='.25';el('video-status').textContent='영상 없음 / 마지막 수신 화면';}}
 finally{setTimeout(()=>pollVideo(annotated),100);}
}
setInterval(()=>{const now=performance.now();if(deadline&&now>=deadline)expire();if(policyDeadline&&now>=policyDeadline){policyDeadline=0;el('policy').textContent='UNKNOWN';el('policy-reason').textContent='safety_decision_expired';}if(videoDeadline&&now>=videoDeadline){videoDeadline=0;el('preview').style.opacity='.25';el('video-status').textContent='영상 없음 / 오래된 프레임';}},50);
document.addEventListener('visibilitychange',()=>{expire('camera_refresh_pending');policyDeadline=0;el('policy').textContent='UNKNOWN';});pollCamera();pollIncident();pollVideo(true);pollVideo(false);
</script></html>'''
