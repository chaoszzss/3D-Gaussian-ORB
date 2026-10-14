"""Keep the actual Gaussian/optimizer allocations within a fixed point budget."""
import torch
from gsplat.strategy import DefaultStrategy
from gsplat.strategy.ops import duplicate,split


class BoundedDefaultStrategy(DefaultStrategy):
    max_gaussians=500000

    @torch.no_grad()
    def _grow_gs(self,params,optimizers,state,step):
        n=len(params['means']);budget=max(0,self.max_gaussians-n)
        if not budget:return 0,0
        gradients=state['grad2d']/state['count'].clamp_min(1)
        high=gradients>self.grow_grad2d
        small=torch.exp(params['scales']).max(-1).values<=self.grow_scale3d*state['scene_scale']
        dupli=high&small;to_split=high&~small
        if step<self.refine_scale2d_stop_iter:to_split|=state['radii']>self.grow_scale2d
        # A small high-gradient splat may also have a large screen footprint.
        # Prefer splitting it, not both operations: each selected original
        # then adds exactly one net GS (duplicate +1; two-child split +1).
        dupli &= ~to_split
        candidates=torch.where(dupli|to_split)[0]
        if len(candidates)>budget:
            chosen=candidates[torch.topk(gradients[candidates],budget).indices]
            selected=torch.zeros(n,dtype=torch.bool,device=candidates.device);selected[chosen]=True
            dupli&=selected;to_split&=selected
        nd=int(dupli.sum());ns=int(to_split.sum())
        if nd:duplicate(params=params,optimizers=optimizers,state=state,mask=dupli)
        to_split=torch.cat([to_split,torch.zeros(nd,dtype=torch.bool,device=to_split.device)])
        if ns:split(params=params,optimizers=optimizers,state=state,mask=to_split,revised_opacity=self.revised_opacity)
        return nd,ns
