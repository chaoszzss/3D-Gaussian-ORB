// CPU regression: source-pinned shader rewrite, independent native references,
// projection closure and one-pass union coverage. Does not create WebGL contexts.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import * as T from '../node_modules/three/build/three.module.js';
import { SparkRenderer } from '../node_modules/@sparkjsdev/spark/dist/spark.module.js';
import { nativeAaShaderSources, installNativeAaShader, rendererConfig, applyNativeCamera } from '../gaussian_direction_support.js';
import { createHash } from 'node:crypto';
const out='reports/viewer_parity_v2_sparse/';
const sm=JSON.parse(fs.readFileSync('viewer/node_modules/@sparkjsdev/spark/dist/spark.module.js.map'));
const shader=name=>JSON.parse(sm.sourcesContent[sm.sources.findIndex(x=>x.endsWith(name))].slice('export default '.length,-1));
const old={vertexShader:shader('splatVertex.glsl'),fragmentShader:shader('splatFragment.glsl')};
const sources=nativeAaShaderSources(old.vertexShader,old.fragmentShader);
assert.throws(()=>nativeAaShaderSources(old.vertexShader.replace('    float clip =','    float clipBROKEN =').replace('if (abs(clipCenter.x) > clip || abs(clipCenter.y) > clip) {','if (false) {'),old.fragmentShader));
assert.equal(rendererConfig({renderer:{schema_version:1,mode:'inria_2dgs_v1'}}).coverage,'spark_3sigma_quad');
assert.equal(rendererConfig({renderer:{schema_version:1,mode:'inria_2dgs_v2'}}).coverage,'spark_3sigma_quad');
assert.equal(rendererConfig({renderer:{schema_version:1,mode:'inria_2dgs_v3'}}).coverage,'native_tiled_aabb_rho_union_v1');
assert.throws(()=>rendererConfig({renderer:{schema_version:1,mode:'inria_2dgs_v3',spark:{maxStdDev:4}}}));
const spark={material:new T.ShaderMaterial({...old}),uniforms:{renderSize:{value:new T.Vector2(640,540)}}};
const diag=installNativeAaShader(spark);assert.equal(diag.passes,1);assert.equal(spark.material.vertexShader,sources.vertexShader);spark.material.dispose();
function project(c) {
 const camera=new T.PerspectiveCamera();applyNativeCamera(T,camera,{camera:c.camera});
 const w2v=camera.matrixWorldInverse;
 const q=new T.Quaternion(...c.quaternion_wxyz.slice(1),c.quaternion_wxyz[0]);
 const invQ=new T.Quaternion().setFromRotationMatrix(w2v);
 const vq=invQ.multiply(q);
 const center=new T.Vector3(...c.center).applyMatrix4(w2v);
 const u=new T.Vector3(c.scales[0],0,0).applyQuaternion(vq);
 const v=new T.Vector3(0,c.scales[1],0).applyQuaternion(vq);
 const clip=x=>new T.Vector4(x.x,x.y,x.z,x===center?1:0).applyMatrix4(camera.projectionMatrix);
 const [a,b,p]=[u,v,center].map(clip);const w=c.camera.w,h=c.camera.h;
 const tw=[a.w,b.w,p.w];
 return [tw.map((n,i)=>w*.5*[a.x,b.x,p.x][i]+(w-1)*.5*n),tw.map((n,i)=>-h*.5*[a.y,b.y,p.y][i]+(h-1)*.5*n),tw];
}
const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
function evaluate(c,tm) {
 const [tu,tv,tw]=tm,[x,y]=c.pixel;
 const den=dot([9,9,-1],tw.map(n=>n*n));
 if(den===0||tw[2]<=.2)return {active:false,alpha:0};
 const f=[9,9,-1].map(n=>n/den);
 const image=[tu,tv].map(t=>dot(f,t.map((n,i)=>n*tw[i])));
 const extent=[tu,tv].map((t,j)=>Math.sqrt(Math.max(.0001,image[j]**2-dot(f,t.map(n=>n*n)))));
 const radius=Math.ceil(Math.max(...extent,3*.707106));
 const grid=[Math.ceil(c.camera.w/16),Math.ceil(c.camera.h/16)];
 const lo=image.map((n,i)=>Math.max(0,Math.min(grid[i],Math.trunc((n-radius)/16))));
 const hi=image.map((n,i)=>Math.max(0,Math.min(grid[i],Math.trunc((n+radius+15)/16))));
 const covered=[x,y].every((n,i)=>n>=lo[i]*16&&n<hi[i]*16);
 const cr=cross(tw.map((n,i)=>x*n-tu[i]),tw.map((n,i)=>y*n-tv[i]));
 const uv=cr.slice(0,2).map(n=>n/cr[2]);const depth=dot(uv,tw.slice(0,2))+tw[2];
 const rho3=dot(uv,uv),rho2=2*((image[0]-x)**2+(image[1]-y)**2);
 const a=Math.min(.99,c.opacity*Math.exp(-.5*Math.min(rho3,rho2)));
 const active=covered&&cr[2]!==0&&depth>=.2&&a>=1/255;
 return {active,alpha:active?a:0,rho3,rho2,depth,image,radius,covered,lo,hi};
}
const cases=JSON.parse(fs.readFileSync(out+'cpu_cases.json'));
const toys=JSON.parse(fs.readFileSync(out+'toy_cases.json'));
let maxT=0,maxAlpha=0,unionOnly=0;
for(const c of [...cases,...toys]) {
 const t=project(c);const err=Math.max(...t.flat().map((v,i)=>Math.abs(v-c.T.flat()[i])));maxT=Math.max(maxT,err);
 assert.ok(err<.001,`projection closure ${err}`);
 const r=evaluate(c,t);
 assert.equal(r.active,c.native_active,JSON.stringify({id:c.id,pixel:c.pixel,r,reference:c}));
 maxAlpha=Math.max(maxAlpha,Math.abs(r.alpha-c.native_alpha));
 assert.ok(Math.abs(r.alpha-c.native_alpha)<.0001,`alpha ${r.alpha} reference ${c.native_alpha}`);
 if(r.active&&r.rho3>9&&r.rho2<r.rho3)unionOnly++;
}
assert.ok(unionOnly>0);
assert.ok(toys.some(c=>c.kind==='tiny_positive'&&c.native_active&&c.rho3d>9));
assert.ok(toys.some(c=>c.kind==='extreme_grazing'&&c.native_active&&c.rho3d>9));
// Same fragment uses min: alpha is max of two falloffs, not sum/over blending.
for(const c of toys.filter(c=>c.native_active)) {
 const r=evaluate(c,c.T),a3=c.opacity*Math.exp(-.5*r.rho3),a2=c.opacity*Math.exp(-.5*r.rho2);
 assert.ok(Math.abs(r.alpha-Math.min(.99,Math.max(a3,a2)))<1e-12);
}
// Expand the same registered shader chunks for CPU glslang syntax/link checking.
function expand(s){return s.replace(/#include <(\w+)>/g,(_,n)=>expand(n==='splatDefines'?shader('splatDefines.glsl'):T.ShaderChunk[n]??(()=>{throw Error(n);})()));}
const vertexPrefix='#version 300 es\nprecision highp float;\nprecision highp int;\nprecision highp sampler2D;\nprecision highp usampler2D;\nprecision highp usampler2DArray;\n#define attribute in\n#define varying out\n#define texture2D texture\nuniform mat4 projectionMatrix;\nuniform bool isOrthographic;\nin vec3 position;\n';
const fragmentPrefix='#version 300 es\n#define varying in\n#define texture2D texture\n#define PREMULTIPLIED_ALPHA\n';
fs.writeFileSync(out+'v3_expanded.vert',vertexPrefix+expand(sources.vertexShader));
fs.writeFileSync(out+'v3_expanded.frag',fragmentPrefix+expand(sources.fragmentShader));
const sha=s=>createHash('sha256').update(s).digest('hex');
const report={status:'cpu_regressions_passed_gpu_pending',actual_contributor_cases:cases.length,toy_cases:toys.length,
 max_projection_T_abs_error:maxT,max_alpha_abs_error:maxAlpha,AA_union_only_cases:unionOnly,
 actual_profile:'inria_2dgs_v3',shader_sha256:{vertex:sha(sources.vertexShader),fragment:sha(sources.fragmentShader)},
 limits:['CPU projection uses float64; actual GPU arithmetic/compilation/image validation pending.','Single primitive union; no second overlapping pass.']};
fs.writeFileSync(out+'regression.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));
