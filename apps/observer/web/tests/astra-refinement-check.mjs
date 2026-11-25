// Safe route-mocked UI regression. No real service writes, game fetches or model runs.
import { chromium } from 'playwright';
import { readFileSync, existsSync, statSync, mkdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';
const root=path.resolve(import.meta.dirname,'..'),dist=process.env.PZ_WEB_DIST||path.join(root,'dist');
const out=process.argv[2]||'/tmp/pitcheezy-astra-ui-check';mkdirSync(out,{recursive:true});
const fixtures=JSON.parse(readFileSync(path.join(root,'tests/fixtures/service-849843.json')));
const browser=await chromium.launch();const report=[];
const check=(v,msg)=>{report.push({pass:!!v,msg});if(!v)console.error('FAIL',msg)};
for(const [w,h] of [[390,844],[390,676],[375,553],[320,568]])for(const which of ['final','live']){
 const body=structuredClone(fixtures[which]);
 for(const p of body.pitches)p.context_before={batter:{at_bats:2,hits:1,on_base:2},pitcher:{pitches:36,strikeouts:2,earned_runs:null},next_batters:[{id:1,name:'Shohei Ohtani',order:1},{id:2,name:'Very Long Mookie Betts Name',order:2}]};
 if(body.current)body.current.context=body.pitches[0].context_before;
 const context=await browser.newContext({viewport:{width:w,height:h},deviceScaleFactor:2});const page=await context.newPage();
 await page.route('http://pz.test/**',r=>{const u=new URL(r.request().url());if(u.pathname.startsWith('/api/'))return r.fulfill({json:body});const f=path.join(dist,u.pathname),file=existsSync(f)&&statSync(f).isFile()?f:path.join(dist,'index.html');return r.fulfill({body:readFileSync(file),contentType:file.endsWith('.js')?'text/javascript':file.endsWith('.css')?'text/css':'text/html'});});
 await page.goto('http://pz.test/watch/849843');await page.waitForSelector('.sv-board');await page.waitForTimeout(250);
 await page.screenshot({path:path.join(out,`${which}-${w}x${h}.png`)});
 const g=await page.evaluate(()=>{const r=s=>document.querySelector(s).getBoundingClientRect(),zone=r('.sv-zone-wrap .sv-zone-box'),rec=r('.sv-recband'),sq=document.scrollingElement;return {dx:zone.x+zone.width/2-innerWidth/2,dy:rec.y-zone.y,zw:zone.width,zh:zone.height,overflow:sq.scrollHeight-innerHeight,horizontal:sq.scrollWidth-innerWidth,bottom:!!document.querySelector('.sv-notice.past'),overlay:!!document.querySelector('.sv-zone-wrap .sv-actual-label'),players:[...document.querySelectorAll('.sv-player')].map(e=>e.children.length),live:document.querySelector('.sv-live').textContent,recOverlap:rec.left<zone.right&&rec.right>zone.left&&rec.top<zone.bottom&&rec.bottom>zone.top};});
 const tiles=await page.locator('.sv-card:not(.pending)').evaluateAll(es=>es.map(e=>({height:e.getBoundingClientRect().height,cap:e.querySelector('.sv-card-cap').getBoundingClientRect().top,zone:e.querySelector('.sv-zone-box').getBoundingClientRect().height})));check(tiles.every(t=>Math.abs(t.height-tiles[0].height)<1&&Math.abs(t.cap-tiles[0].cap)<1&&Math.abs(t.zone-tiles[0].zone)<1),'uniform tile grid and captions '+which+w+h);
 report.push({which,w,h,geometry:g});check(Math.abs(g.dx)<1,'center '+which+w+h);check(g.horizontal<=1,'no horizontal '+which+w+h);check(g.overflow<=1,'no page scroll '+which+w+h);check(!g.recOverlap,'rec beside zone '+which+w+h);check(Math.abs(g.dy)<1,'rec aligned '+which+w+h);check(!g.bottom&&!g.overlay,'no oldbar/pointlabel');check(g.players.every(x=>x===4),'four rows');
 if(which==='final')check(g.live==='종료','final state');
 await page.locator('.sv-inning').click();check(await page.locator('.sv-pa-controls select').count()===1,'PA selector');await page.getByRole('button',{name:'박스스코어',exact:true}).click();check(await page.getByRole('dialog').count()===1,'box sheet');await page.keyboard.press('Escape');await page.locator('.sv-player.away').click();check((await page.locator('.sv-sheet').textContent()).includes('오늘 기록'),'player details');await page.keyboard.press('Escape');
 if(which==='live'){await page.locator('.sv-strip .sv-card:not(.pending)').first().click();check((await page.locator('.sv-live').textContent())==='최신으로 ↗','pinned latest action');await page.locator('.sv-live').click();check((await page.locator('.sv-live').textContent())===(body.source?.session_id?'시연 중':'LIVE'),'latest state');}
 await context.close();
}
// Overflow indicators and record/clock state regressions use synthetic fixture values only.
for (const mode of ['many','paused','no-data','outside']) {
 const body=structuredClone(fixtures.final);
 if(mode==='many') {
  const first=body.pitches[0]; body.pitches=Array.from({length:12},(_,i)=>({...structuredClone(first),index:i,pitch_number:i+1,key:`test:${i}`}));
 } else if(mode==='paused') { Object.assign(body,structuredClone(fixtures.live));body.source.session_status='paused'; }
 else if(mode==='outside'){body.pitches[0].batter.side='L';body.pitches[0].actual.x=-1.25;body.pitches[0].actual.z=3.15;}
 else {body.pitches[0].rec=null;body.pitches[0].actual=null;}
 const context=await browser.newContext({viewport:{width:mode==='outside'?320:390,height:mode==='outside'?568:676}});const page=await context.newPage();
 await page.route('http://pz.test/**',r=>{const u=new URL(r.request().url());if(u.pathname.startsWith('/api/'))return r.fulfill({json:body});const f=path.join(dist,u.pathname),file=existsSync(f)&&statSync(f).isFile()?f:path.join(dist,'index.html');return r.fulfill({body:readFileSync(file),contentType:file.endsWith('.js')?'text/javascript':file.endsWith('.css')?'text/css':'text/html'});});
 await page.goto('http://pz.test/watch/849843');await page.waitForSelector('.sv-board');await page.waitForTimeout(200);
 if(mode==='many'){
  const edges=()=>page.locator('.sv-strip-shell').getAttribute('class');
  check((await edges()).includes('fade-right')&&!(await edges()).includes('fade-left'),'strip first edge');
  await page.locator('.sv-strip').evaluate(el=>el.scrollLeft=(el.scrollWidth-el.clientWidth)/2);await page.waitForTimeout(150);
  check((await edges()).includes('fade-left')&&(await edges()).includes('fade-right'),'strip middle edges');
  await page.locator('.sv-strip').evaluate(el=>el.scrollLeft=el.scrollWidth);await page.waitForTimeout(150);
  check((await edges()).includes('fade-left')&&!(await edges()).includes('fade-right'),'strip final edge');
  await page.locator('.sv-strip .sv-card').last().click();await page.waitForTimeout(100);
  const visible=await page.locator('.sv-strip .sv-card').last().evaluate(el=>{const r=el.getBoundingClientRect(),s=el.parentElement.getBoundingClientRect();return r.left>=s.left&&r.right<=s.right;});check(visible,'selected final card fully visible');
  await page.locator('.sv-player.away').click();await page.keyboard.press('Tab');await page.keyboard.press('Shift+Tab');check(await page.locator('.sv-sheet').evaluate(el=>el.contains(document.activeElement)),'sheet focus trapped');await page.keyboard.press('Escape');check(await page.locator('.sv-player.away').evaluate(el=>el===document.activeElement),'player focus restored');
  await page.locator('.sv-inning').click();await page.getByRole('button',{name:'박스스코어',exact:true}).click();await page.keyboard.press('Escape');await page.waitForTimeout(100);check(await page.locator('.sv-inning').evaluate(el=>el===document.activeElement),'nested sheet opener restored');
 } else if(mode==='paused') {check(await page.locator('.sv-live').textContent()==='일시정지','paused demo distinguished');check(await page.locator('.sv-zone-wrap .sv-actual,.sv-zone-wrap .sv-mitt').count()===0,'prepitch hides actual and mitt');}
 else if(mode==='outside'){check(await page.locator('.sv-zone-wrap .sv-actual circle').evaluate(el=>{const r=el.getBoundingClientRect(),b=document.querySelector('.sv-recband').getBoundingClientRect();return !(r.left<b.right&&r.right>b.left&&r.top<b.bottom&&r.bottom>b.top);}),'outside actual visible over no recommendation text');}
 else {check(await page.locator('.sv-actual-summary').textContent()==='실제 투구 기록 없음','missing actual explicit');check(await page.locator('.sv-upcoming').textContent()==='–번 · 타자 미확인–번 · 타자 미확인','missing lineup stays unknown');check((await page.locator('.sv-player.away .sv-player-3').textContent()).includes('–타수'),'missing stats stay unknown');}
 await page.screenshot({path:path.join(out,mode+'.png')});await context.close();
}
await browser.close();writeFileSync(path.join(out,'checks.json'),JSON.stringify(report,null,2));console.log(JSON.stringify({checks:report.filter(x=>'pass'in x).length,failures:report.filter(x=>x.pass===false).length}));process.exitCode=report.some(x=>x.pass===false)?1:0;
