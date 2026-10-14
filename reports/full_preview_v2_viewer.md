# 全场拼接预览：独立用户入口与查看器验收

[统一入口](http://127.0.0.1:8767/reports/full_preview_v2/) · [全场总览](http://127.0.0.1:8769/) · [原始分区详图](http://127.0.0.1:8770/)

主代理已启动并持有这些用户服务。验收脚本只连接已有服务，只清理自己启动的浏览器，不关闭用户8768浏览器或任何用户服务。

本次使用26份模型资产覆盖原28区域：000换最新measuredseed＋切向尺度上限退火15k模型；014/015/016由一份joint015联合模型替换；其余24个区域保留历史模型。完整PLY保留全部13,681,252条记录；总览为1,200,000点真实分层抽样，允许实际采样包围盒内缩。楼层暂定，混合模型未通过高保真验收。

2026-10-09实际验收结果：**VIEWER_SMOKE_PASS**。这是加载、资产身份、视域和区域导航检查；混合模型仍为 `high_fidelity_accepted=false`，未进行新native parity、整图质量评分或全模型同时加载。

| 实际检查 | 结果 |
| --- | --- |
| CPU服务身份 | 26份详图＋1份总览，27条模型路由均HEAD 200／Range 206；实际本地PLY SHA与assembly及public manifest一致 |
| 总览浏览器加载 | 1,200,000个高斯，`inria_2dgs_v3`；加载文件SHA `d3df427141a27568e6812733467d48d465891754bb93613aabc58be4f2d5ec18` |
| 总览视域 | 实际斜视、侧视相机矩阵分别投影完整bbox的8个角，均在视锥内；仅证明相机覆盖完整坐标范围 |
| 楼层导航 | UI含F1–F6及坡道索引；实际选择F6，列出025／026／027 |
| F6实际加载 | 历史 `runs/garage_surface_v1/block_027/eval_dev/gaussians.ply`，235,743点，加载SHA与原资产相同；通过“进入区域”切到既有视角2 |
| 渲染错误 | 页面／console错误0；三张截图处WebGL `getError()=0`，context未丢失 |
| GPU与生命周期 | NVIDIA RTX 4060 Laptop GPU；独占共享锁39.35秒，全程持锁，自己的浏览器全部关闭后释放 |
| 资源采样 | 全板显存起始1385／最高2071／结束1134 MiB；72次采样，包含用户进程，不能当作本模型独占峰值 |
| 用户服务 | 测试后8767、8768、8769、8770全部HTTP 200；未启动或关闭用户server，未接管用户浏览器 |

截图均为1440×960真实浏览器窗口，实际模型画布1120×710：

- [全场斜视](full_preview_v2/overview.png)
- [全场侧视](full_preview_v2/overview_side.png)
- [F6 block027参考视角](full_preview_v2/f6_block_027.png)

目视可见总览的整体分布与高程起伏；完整bbox适配较保守，截图主体偏小，可在用户查看器中滚轮放大。F6参考图确实显示室内车库内容，仍有模糊、颜色溢出及局部表面缺陷，不能认定高保真达标。米制网格在本次截图中关闭，无LiDAR叠加。

完整bbox（米）为 `[-38.333412,-71.572182,-12.985082]` 至 `[104.465080,80.498337,21.970640]`。总览实际采样bbox为 `[-38.333412,-71.101669,-11.911044]` 至 `[104.465080,80.288818,21.793308]`，允许内缩，未宣称逐极值保留。源覆盖、拼接属性与记录保存审计复用[独立源审计](full_preview_v2_source_audit.json)及[assembly](../maps/garage_fidelity_v2_full_preview/assembly.json)；本轮没有重复全量质量评估。

用户服务已由主代理启动，直接访问上方链接即可。8769打开后点击总览卡片“加载”；8770选择楼层再加载原始块，可用“进入区域”进入既有参考视角。两个约250万点的新模型切换前先卸载，隐藏仍占预算。8769为默认150万点／512 MiB／3块；8770为400万点／640 MiB／4块，文件内存预算也限制同时加载数量。

若用户服务已停止，以下是在工程根目录分别启动两个查看器的原命令；服务正在运行时无需重复执行：

```bash
python3 -B scripts/serve_map.py --port 8769 --manifest maps/garage_fidelity_v2_full_preview/overview_manifest.json
python3 -B scripts/serve_map.py --port 8770 --manifest maps/garage_fidelity_v2_full_preview/manifest.json --max-splats 4000000 --max-file-bytes 671088640 --max-loaded-blocks 4
```

本轮独立写入：`reports/full_preview_v2/index.html`、本报告及[JSON明细](full_preview_v2_viewer.json)、上述3张截图、`viewer/tests/full_preview_v2_browser.py`／`.mjs`。harness只连接已有8769／8770，带110秒截止与仅清理自身后代进程的保护；不修改app、shader、server、模型或历史报告。已完成的smoke默认禁止重复执行。JSON保留资产hash、实际相机矩阵、模型计数、截图hash、显存采样及锁生命周期。
