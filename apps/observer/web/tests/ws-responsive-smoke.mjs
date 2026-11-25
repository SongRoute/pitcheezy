// Entirely synthetic rows; no source game, model, network or live backend.
import { chromium } from 'playwright';
import { readFileSync, existsSync, statSync, mkdirSync } from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
const dist = path.resolve('dist'), shots = path.resolve(process.argv[2] ?? '/tmp/pitcheezy-ws-ui-1013'); mkdirSync(shots, {recursive:true});
const situation = {inning:1,half:'Top',balls:1,strikes:1,outs:2,bases:1,home_score:0,away_score:0};
const person = {id:11,name:'합성 타자',side:'R'}, pitcher = {id:22,name:'합성 투수',hand:'R'};
const rec = {status:'ready',provenance:'captured_live',candidates:[{rank:1,pitch_type:'SI',pitch_label:'싱커',zone_label:'낮은 왼쪽',zone_id:'low_left',target:{x:-.55,z:1.7},probability:.5},{rank:2,pitch_type:'FC',pitch_label:'커터',zone_label:'가운데 오른쪽',zone_id:'mid_right',target:{x:.55,z:2.5},probability:.3},{rank:3,pitch_type:'FF',pitch_label:'포심',zone_label:'높은 가운데',zone_id:'high_middle',target:{x:0,z:3.5},probability:.2}]};
const context={batter:{at_bats:0,hits:0,on_base:0},pitcher:{pitches:2,strikeouts:0,earned_runs:0},next_batters:[],lineups:{away:[{id:11,name:'합성 타자',order:1}],home:[]},boxscore:{away:{batters:[],pitchers:[]},home:{batters:[],pitchers:[]}}};
const pitch=(index)=>({index,key:`999:1:${index+1}`,pa_key:'999:1',at_bat_number:1,pitch_number:index+1,situation_before:situation,batter:person,pitcher,rec,receipt:{status:'received',corrected:false},actual:{pitch_type:index?'SI':'FC',pitch_label:index?'싱커':'커터',speed_mph:90,x:index?.5:.2,z:index?1.3:1.7,result_label:index?'볼':'파울',description:index?'ball':'foul',event:null,catcher_setup:{plate_x_feet:.3}},context_before:context});
const make=()=>({game:{game_pk:999,date_kst:'2026-10-13',start:'2026-10-13T07:00:00Z',status:'live',away_team:'Synthetic Away',home_team:'Synthetic Home',away_score:0,home_score:0,inning:1,half:'Top'},source:{kind:'live',session_id:null},feed:{ok:true},zone_bounds:{bottom:1.5,top:3.5},pitches:[pitch(0),pitch(1)],current:{phase:'pre_pitch',situation,batter:person,pitcher,pa_key:'999:1',context},next:{rec},cutoff:{index:1},pas:[{pa_key:'999:1',at_bat_number:1,inning:1,half:'Top',batter:'합성 타자',pitcher:'합성 투수',first:0,last:1,complete:false,result:'타석 진행 중',pitch_count:2}],display_state:{schema:'pitcheezy-display-state-v1',state:'recommended',reason:'ledger_recorded',label:'다음 공 추천',pitch_key:'999:1:3',recommendation_shown:true,feed:{ok:true,delayed:false,lag_s:null}},win_expectancy:{status:'available',home_probability:.6,perspective:'home'},notes:[]});
const browser=await chromium.launch();
for(const [width,height] of [[1440,1000],[390,844],[320,568],[1024,768]]){
 const ctx=await browser.newContext({viewport:{width,height}}), page=await ctx.newPage();let body=make(), polls=0, down=false;const errors=[];page.on('pageerror',e=>errors.push(String(e)));
 await page.route('http://pz.test/**',route=>{const u=new URL(route.request().url());if(u.pathname.startsWith('/api/')){polls++;if(down)return route.abort('internetdisconnected');return route.fulfill({json:body});}let file=path.join(dist,u.pathname);if(!existsSync(file)||!statSync(file).isFile())file=path.join(dist,'index.html');return route.fulfill({body:readFileSync(file),contentType:file.endsWith('.js')?'text/javascript':file.endsWith('.css')?'text/css':'text/html'});});
 await page.goto('http://pz.test/watch/999');await page.locator('.sv-rec-row').first().waitFor();
 assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true,`${width} horizontal overflow`);
 await page.screenshot({path:`${shots}/${width}-live.png`,fullPage:true});
 assert.equal(await page.locator('.sv-stage .sv-pa-dot').count(),2);
 if(width===390)assert.ok((await page.locator('.sv-strip').boundingBox()).y+(await page.locator('.sv-strip').boundingBox()).height <= (await page.locator('.sv-bottom-nav').boundingBox()).y,'390 essential pitches above nav');
 const we=page.locator(width<760?'.sv-mobile-we':'.sv-game-context .sv-panel-link').filter({hasText:'홈팀 승리확률'});assert.match(await we.innerText(),/60%/);await we.click();assert.match(await page.locator('.sv-sheet').innerText(),/60%/);down=true;await page.waitForTimeout(2200);assert.doesNotMatch(await page.locator('.sv-sheet').innerText(),/60%/);down=false;await page.getByRole('button',{name:'닫기',exact:true}).click();await page.waitForTimeout(2200);
 await page.locator('.sv-strip button').first().click();await page.locator('.sv-sheet').waitFor();
 assert.equal(await page.locator('.sv-sheet').getAttribute('aria-modal'),width>=1200?null:'true');
 const before=polls; body={...body,pitches:[{...pitch(0),index:1,receipt:{status:'received',corrected:true},actual:{...pitch(0).actual,speed_mph:91}},{...pitch(1),index:2},{...pitch(2),index:3}],cutoff:{index:3},current:{...body.current,situation:{...situation,balls:2}},display_state:{...body.display_state,pitch_key:'999:1:4'}};
 await page.waitForTimeout(2400);assert.ok(polls>before,'polling continues with detail open');assert.match(await page.locator('.sv-sheet').innerText(),/정정됐어요/);assert.match(await page.locator('.sv-state-head').innerText(),/4구/);assert.match(await page.locator('.sv-sheet').innerText(),/1구/);
 await page.screenshot({path:`${shots}/${width}-pitch-detail.png`,fullPage:true});
 if(width>=1200){await page.locator('.sv-panel-link').filter({hasText:'이닝별 점수표'}).click();assert.match(await page.locator('.sv-sheet').innerText(),/이닝별 기록/);}
 await page.screenshot({path:`${shots}/${width}-detail.png`,fullPage:true});await page.getByRole('button',{name:'닫기',exact:true}).click();
 body={...body,pitches:body.pitches.map(p=>p.key==='999:1:1'?{...p,receipt:{status:'unconfirmed',corrected:true}}:p)};await page.waitForTimeout(2200);await page.locator('.sv-strip button').filter({hasText:'1. 확인 중'}).click();assert.match(await page.locator('.sv-sheet').innerText(),/다시 확인/);assert.equal(await page.locator('.sv-sheet .sv-table tbody tr').count(),0);assert.doesNotMatch(await page.locator('.sv-sheet').innerText(),/147km/);await page.getByRole('button',{name:'닫기',exact:true}).click();
 for(const state of ['unsupported','feed_delayed','unconfirmed','disconnected','ledger_unavailable','waiting','unavailable','pregame','final']){body={...body,display_state:{...body.display_state,state,label:`검사 ${state}`,message:'합성 상태 안내',recommendation_shown:false,feed:{ok:true,delayed:state==='feed_delayed',lag_s:52}}};await page.waitForTimeout(2150);assert.equal(await page.locator('.sv-state').getAttribute('data-state'),state);assert.equal(await page.locator('.sv-stage .sv-rec').count(),0,`hidden recommendation ${state}`);}
 assert.deepEqual(errors,[]);console.log(`PASS ${width}x${height}: layout, key correction, live polling, dialog, 10 display states`);await ctx.close();
}
await browser.close();
