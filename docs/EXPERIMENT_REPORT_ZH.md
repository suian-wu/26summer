# 实验报告：基于 VLM+SAM 的语言指令抓取放置策略（graspbench）

版本：v2（补充消融/失败分析细节，修正 v62/v63 因果标注）
提交策略文件：`policies/student_policy.py`（对应 `right_policy_and_record/` 中记录的 `task123_sort_v66` 迭代结果）

---

## 1. 问题定义与边界

### 1.1 任务

在 MuJoCo 仿真环境中，机械臂需根据自然语言指令（中/英文混合），识别桌面上的目标物体与目标容器，完成抓取并放置。共三个任务：

- **Task1（基础形状放置）**：物体为 `red_cube`、`green_cylinder`、`blue_box` 三种规则几何体，容器为 `square_tray`（方盘）、`round_tray`（圆盘）。6 个公开 episode，seed 31–36，每次场景中三个物体同时存在，逐一考察不同物体-容器配对。
- **Task2（YCB 物体放置）**：物体为 `banana`、`apple`（类别 `fruit`），场景固定为 banana→round_tray、apple→square_tray。6 个公开 episode，seed 61–66，主要变化是指令的语言与措辞（中/英文、主动/被动句式）。
- **Task3（多物体分拣）**：物体为 `apple`、`orange`（水果类，→round_tray）与 `mustard_bottle`、`potted_meat_can`（包装食品类，→square_tray），需要一条指令下完成 4 组抓取-放置。3 个公开 episode，seed 91–93。

### 1.2 观测与动作

- **观测**：双相机 RGB-D（`overhead` 俯视 + `front` 前视，256×256），机器人关节位置/速度，末端位姿（`ee_position`/`ee_quaternion`），夹爪开度；相机外参（`position`/`rotation`）与内参（`fovy_degrees`）随观测一起给出。
- **动作**：仅支持关节位置指令（`JointPositionCommand`，7 个机械臂关节角 + 1 个夹爪开度），不支持直接下发笛卡尔位姿（`CartesianDeltaCommand` 需先经 `CartesianDeltaAdapter` 转换为关节角）。控制频率由 `control_substeps=20` 与物理时间步共同决定。

### 1.3 成功、失败与安全约束

- **成功判定（策略不可见，仅评测器内部使用）**：目标物体中心 XY 落在容器内半径/半边长范围内，Z 高度在桌面到最大堆叠高度之间，且线速度、夹爪开度均满足阈值，并需要**连续 10 个控制步**同时满足才计为该目标成功；Task3 需要 4 组目标-容器配对**同时**连续 10 步满足才算整回合成功。策略自身在日志中记录的 `verified`/`done` 只是自我判断，不能代表评测器认定的物理成功。
- **失败/终止条件**：控制预算耗尽（Task1/2 为 800 步，Task3 为 3000 步）、模型服务持续报错、策略主动进入安全停止。
- **安全约束**：手部/手指/夹爪垫片与桌面、桌腿、地面之间的接触计为“危险接触”（unsafe contact），逐步计数并汇总进 `summary.json`。
- **边界限制**：策略只能使用公开指令文本、当次 `Observation` 以及只读机器人模型；不能读取隐藏目标、随机种子或仿真真值作为决策输入。目标位置必须来自真实视觉模型输出的 mask/box 与当前 RGB-D 反投影，不允许硬编码坐标或直接查表。

### 1.4 外部依赖声明

- **VLM（语言理解与任务规划）**：通过 OpenAI 兼容的 `/chat/completions` 协议调用外部/自建服务（`GRASPBENCH_VLM_BASE_URL`，模型名默认 `qwen3-vl-flash`），用于将指令解析为 pick/place 目标 ID 序列，返回结果限定在 SAM 已检测到的候选范围内，不允许模型“发明”目标。
- **检测服务**：自建 HTTP 服务（`scripts/sam_server.py`），协议接口命名沿用 “SAM3”（如 `GRASPBENCH_SAM3_URL`、`service: "sam3"`），但**实际模型组合是 Grounding-DINO（`IDEA-Research/grounding-dino-base`）做开放词汇目标检测 + SAM2（`sam2.1_hiera_large`）做像素级分割 + OWLv2（`google/owlv2-base-patch16-ensemble`）作为 Grounding-DINO 检测失败时的救援检测器**，并非 Meta 官方发布的 SAM 3 模型。这一点在代码注释和协议命名中容易引起误解，特此说明。
- **预训练权重**：Grounding-DINO、SAM2、OWLv2 均为公开预训练权重，未做任何微调（fine-tune），仅用作现成的开放词汇检测/分割/兜底检测组件。
- **未使用额外数据**：训练/评测所用的物体、容器、场景布局均由仿真环境按配置文件与 seed 生成，未引入仿真外的额外数据集。

---

## 2. 系统架构

### 2.1 信息流与控制频率

```
                    ┌────────────────────────┐
   指令(text) ─────▶│   VLM（远程/自建服务）   │──▶ pick_id/place_id 或
                    │  plan_task/ground_target│    完整 actions 序列
                    └────────────────────────┘
                              ▲
                              │ 候选目标 ID 列表（来自检测结果，VLM 不能越界选择）
                              │
   RGB-D(overhead+front) ─┐   │
                          ▼   │
                    ┌────────────────────────┐
                    │  检测服务 scripts/      │
                    │  sam_server.py (本地)   │──▶ boxes/scores/masks(位打包)
                    │  DINO→(失败)→OWLv2→SAM2 │
                    └────────────────────────┘
                              │
                              ▼
                    ┌────────────────────────┐
                    │ perception.py：反投影   │──▶ 物体世界坐标 + 朝向估计(yaw)
                    │ （像素+深度→相机系→世界系）│
                    └────────────────────────┘
                              │
                              ▼
                    ┌────────────────────────┐
                    │ IK（DampedLeastSquaresIK）─▶ 关节目标 q_target
                    └────────────────────────┘
                              │
                              ▼
                    ┌────────────────────────┐
                    │ 状态机（StudentPolicy）  │──▶ JointPositionCommand
                    │ 抬升/平移/对准/下降/闭合/│    （7 关节角 + 夹爪开度）
                    │ 抬升/运输/下降/释放/验证 │
                    └────────────────────────┘
                              │
                              ▼
                     MuJoCo 物理仿真（20 个 substep / 控制步）
```

控制步与仿真子步的比例固定为 `control_substeps=20`；VLM/检测服务调用发生在状态切换的关键节点（如首次感知、目标切换、验证失败重试），而不是每个控制步都调用，以降低远端延迟对实时控制的影响。

### 2.2 模块清单（输入/输出 schema、运行位置、版本、超时、回退）

| 模块 | 运行位置 | 版本/模型 | 输入 | 输出 | 超时 | 失败回退 |
|---|---|---|---|---|---|---|
| VLM（`src/graspbench/vlm.py`） | 远程/自建，OpenAI 兼容接口 | 默认 `qwen3-vl-flash`（`.env` 可覆盖） | 指令文本 + base64 JPEG 图像 + 候选目标 ID 列表 | JSON：`{target_id, reason}`（ground_target）或 `{actions:[{pick_id,place_id}], reason}`（plan_task） | `GRASPBENCH_VLM_TIMEOUT_S`，默认 45s | 抛 `ModelServiceError`，错误信息内嵌原始响应片段用于诊断；策略侧按重试/安全停止处理，不允许臆造目标坐标 |
| 检测服务（`scripts/sam_server.py`） | 本机 HTTP 服务，默认 `127.0.0.1:8765` | Grounding-DINO-base + SAM2.1-hiera-large + OWLv2-base-patch16-ensemble | JPEG(base64) + 文本提示词列表 | `{results:[{scores, boxes(xyxy), masks:{count,packed}}]}` | 客户端 `GRASPBENCH_SAM3_URL` 调用默认 30s | Grounding-DINO 未检出可用候选（含超大框剔除后）时自动降级到 OWLv2（阈值单独校准为 0.10，因其置信度量级远低于 DINO 的 0.30） |
| 几何/坐标（`src/graspbench/perception.py`） | 本地（策略进程内） | — | mask/box + 深度图 + 相机内外参 | 物体世界坐标、朝向四元数、置信度 | 无网络调用，纯计算 | 检测框过大（占画面比例超阈）时在服务端已被剔除；朝向估计失败时退化为不旋转（直立抓取） |
| IK（`src/graspbench/ik.py`） | 本地 | `DampedLeastSquaresIK`，`damping=1e-3, step_size=0.65, orientation_weight=0.35` | 目标位置+姿态、当前关节角 | 关节角增量 | 无网络；`max_iterations` 可由调用方覆盖（默认 180，某些阶段用到 320/420） | 未收敛时按当前最优迭代结果返回，由状态机的到达容差与超时逻辑决定是否重试/换路径 |
| 状态机（`policies/student_policy.py`） | 本地 | `StudentPolicy` | `Observation` | `JointPositionCommand` | 无网络；受控制预算约束 | 抓取/验证失败时进入受限次数的重试（`MAX_RETRIES_PER_ACTION`），耗尽后安全停止而非强行判定成功 |

### 2.3 异步执行相关参数

- 默认执行模式：`async` + `realtime`，评测器单 worker（同一时刻至多 1 个 `act()` 推理在途）。
- **命令复用窗口**：默认最多复用同一条推理结果 8 个控制步（对应约 0.32s @ 25Hz），由 `--async-command-hold-steps` 可调。
- **安全保持条件**：末端执行器发生 >2.5cm 位移或任一关节角变化 >0.15rad，视为旧推理结果过期（stale），转入 `safe_hold`，不再执行过期命令直到新推理返回。
- **状态过期门限**：见上，位移/关节角变化任一超阈即触发。
- **延迟分布**：从 `right_policy_and_record` 的实测日志采样（如 `task3_t002_verify_v66/summary.json`）：单个 episode 内 `async_control.last_latency_s≈0.31s`，`submitted=1171` 次推理请求中 `accepted=1170`、`stale_dropped=1`、`safe_hold_steps=40`、`reused_command_steps=1231`（约占该 episode 总步数 2441 的一半），说明命令复用机制在高延迟下承担了相当比例的控制步。

---

## 3. 方法

### 3.1 目标 grounding

两级流程：
1. **检测（是什么/在哪）**：`sam_server.py` 先用 Grounding-DINO 按硬编码提示词（如 `"square tray"`、`"red cube"`）做开放词汇框检测；剔除占画面面积超过 `MAX_BOX_AREA_FRACTION=0.40` 的框（防止把整张桌面误判为托盘）；若该提示词下 Grounding-DINO 一无所获，自动切换到 OWLv2（独立校准阈值 0.10）重试；确定框后再用 SAM2 生成像素级掩码。
2. **规划（该抓哪个/放哪里）**：VLM 只能在检测服务返回的候选 ID 集合中选择 `pick_id`/`place_id`，不能自创目标，其输出经服务端 schema 校验（ID 必须存在于候选或物体/容器规格表中）后才被接受。

### 3.2 真实模型 endpoint 与请求/响应 schema

- VLM 请求体：`{model, messages:[{role, content:[{type:"text",...},{type:"image_url",...}]}], response_format:{"type":"json_object"}}`（可经 `GRASPBENCH_VLM_RESPONSE_FORMAT=none` 关闭 JSON 强制模式）。响应内容为纯文本 JSON，解析失败时异常消息中会截断嵌入原始响应，便于排查“模型没有按格式回复”与“网络/鉴权错误”两类问题。
- 检测请求体：`{image_jpeg_b64, prompts:[str,...]}` → `POST /infer`；响应 `{results:[{scores:[float], boxes:[[x1,y1,x2,y2]], masks:{count,packed(base64位打包布尔掩码)}}]}`；另有 `GET /healthz` 供启动自检。

### 3.3 mask/box/point 证据与坐标/几何

- 证据类型：box（用于快速筛选候选、防止误检占用过大面积）、mask（像素级，用于精确定位物体中心与朝向）。
- 坐标反投影：用相机 `fovy_degrees` 现算等效焦距（假定正方形像素、主点在图像中心），结合深度图将像素点反投影到相机坐标系，再经 `camera.position + points_camera @ camera.rotation.T` 转换到世界坐标系。
- 朝向估计：用旋转最小面积包围盒扫描（非 PCA）估计物体在桌面上的偏航角；根据长宽比自动判断旋转周期是 90°（近似正方形，如 `red_cube`/`blue_box`）还是 180°（细长形，如 `banana`），仅当估计置信度超过阈值（`ELONGATION_ALIGN_THRESHOLD=0.25`）时才采用该朝向对齐夹爪，否则维持默认直立抓取姿态，避免误估计导致的抓取角度错误。

### 3.4 IK/规划

数值迭代式逆解 `DampedLeastSquaresIK`（阻尼最小二乘），面向单一 `grasp_site`，同时约束位置误差与姿态误差（`orientation_weight=0.35`），支持零空间姿态偏置使多解情形下倾向选取靠近参考姿态的解，且每步关节增量裁剪，避免因残差突变导致的关节速度冲击。规划层是显式有限状态机（非学习式 VLA），阶段包括：起始位移避让视野遮挡 → 初始化/感知 → 抬升 → 平移运输 → 对准朝向 → 下降 → （settle）→ 闭合 → 抬升 → 运输至目标容器上方 → 下降 → 释放 → 抬升 → （settle）→ 验证 → 完成/验证失败重试/模型错误安全停止。

对于水平距离超出竖直下探可达范围的目标（Task3 的 `mustard_bottle` 等远处包装食品），策略引入**前倾抓取**：判定目标是否超出可达距离阈值后，将默认竖直抓取姿态绕水平轴旋转一个固定倾角，使夹爪呈前倾姿态接近目标，用倾斜换取更大的有效水平可达范围，而不是依赖笛卡尔路径规划或多关节协同摆动。

### 3.5 夹爪控制

夹爪开度由控制向量的第 8 维直接给出（`gripper_opening*255.0` 映射到执行器控制量），策略在闭合/释放阶段各设置固定等待时长（如闭合后等待若干秒确认稳定夹持），并结合视觉重检测与夹爪开度反馈共同判断“是否已抓稳”/“是否已释放”，而非只信任单一信号。

### 3.6 动作后验证与恢复

放置后不立即判定完成，而是先进入短暂稳定等待（settle）再进入验证阶段读取当前物体位置/速度，减少物体刚放入托盘时的残余摆动导致的误判；验证失败时按有限次数（`MAX_RETRIES_PER_ACTION=2`）重新感知与重新规划，超出重试预算或模型持续报错则安全停止，不强行宣称成功。

### 3.7 模型失败回退

- VLM 不可用/超时：`right_policy_and_record` 记录的历史迭代中曾实现“VLM 不可用时仍可通过 SAM 检测结果+规则解析完成单物体指令”的回退路径；当前提交版本的回退策略是有限重试后安全停止，不臆造坐标、不读取仿真真值兜底。
- 检测服务不可用：同样进入有限重试与安全停止流程，不使用固定/历史坐标代替实时检测结果。

---

## 4. 实验设计

### 4.1 数据切分

仅使用仓库提供的公开评测集（`configs/task1_place_public.json`、`task2_ycb_public.json`、`task3_sort_public.json`），未见到独立的自建集或隐藏集在本地被使用/生成的记录；隐藏目标信息（`hidden_goals`/`expected_target`）只出现在评测输出的 `summary.json` 中供事后核验，运行时对策略不可见。

### 4.2 Seed 与重复次数

| 任务 | Episode 数 | Seed | 控制预算 |
|---|---|---|---|
| Task1 | 6 | 31–36 | 800 步 |
| Task2 | 6 | 61–66 | 800 步 |
| Task3 | 3 | 91–93 | 3000 步 |

容器位置额外用 `seed+8381` 派生，与物体位置的随机性来源分离。每个 seed 对应固定的物体/容器摆放与指令文本，未做同一 seed 多次重复采样以统计方差（当前实验设计下每个 episode 只跑一次）。

### 4.3 冻结项

- 物理参数（摩擦、质量、碰撞几何）、成功判定阈值、任务配置文件、机器人模型：全程未修改。
- 允许修改范围：策略的位置目标、姿态目标、速度曲线、状态机结构、重试与恢复逻辑，以及检测服务侧的框筛选/级联逻辑（不修改 VLM/SAM 的硬编码提示词文本）。

### 4.4 主要指标

`success_rate`、`target_selection_accuracy`、`verification_rate`、`mean_control_steps`、`mean_wall_time_s`、`total_unsafe_contacts`、`mean_non_target_displacement_m`、`goal_completion_rate`（按目标数而非按 episode 数加权，用于衡量 Task3 部分完成情况）、`task_plan_match_rate`。

### 4.5 Baseline 与消融

- **Baseline**：项目自带的零动作模板（早期 `runs/public_eval` 记录的 `success_rate=0.0`、平均 77.3 步即耗尽判定为失败，用于确认“加载了真实策略文件”而非模板占位）。
- **消融/对照（体现在版本迭代中，非严格对照实验，即非同一批次固定其余变量的 A/B 测试，而是时间序列上的连续修改）**：
  - 检测级联：仅 Grounding-DINO vs. Grounding-DINO+OWLv2 级联（后者用于解决前者对方盘等目标的漏检问题）。**说明**：这一项目前只有 `scripts/sam_server.py` 代码注释与 changelog 的文字描述支撑，仓库中未找到独立保存的检出率数字对比文件（诊断脚本 `debug_owlv2_no_threshold.py` 等只向标准输出打印候选框分数，不落盘保存结果），因此报告中不能给出“DINO 单独检出率 vs 级联后检出率”的量化百分比。
  - 抓取姿态：直立抓取 vs. 基于朝向估计的旋转对齐抓取（用于解决斜放物体抓取脱落问题）。
  - 远处目标：直立下探 vs. 前倾抓取（用于解决 Task3 远处包装食品因竖直可达范围不足而无法下探的问题）。
  - 释放判据：单边开度阈值（`opening>0.97` 即判定释放完成）vs. `release_unjam` 双向判据（开度需回落到 `0.97~1.003` 且关节速度 `<0.10` 才判定完成）——这一项**有完整的逐控制步事件日志可交叉核对**，是本报告中证据链最完整的一组消融，详见 5.4 与 6.1 节。
  - 未采用的方案（详见第 7 节讨论）：放宽成功阈值、增大任务预算、读取仿真真值兜底——均被明确排除，不计入结果。

### 4.6 消融证据的可信度分级

由于历史版本的迭代记录并非按严格对照实验设计保存（部分版本只有文字变更记录、部分只有部分 episode 的运行文件），本报告统一按以下三级标注每一处消融/对比数据的可信度，避免混淆“有文件可核实”和“仅有描述”两类证据：

- **A 级（硬数据）**：有 `aggregate.json` 或 `summary.json` 原始文件可直接核实的指标数字。
- **B 级（部分核实）**：有原始事件日志（`events.jsonl`）但缺少汇总文件，或只有单个/部分 episode 的记录，不能代表完整任务集的表现。
- **C 级（仅文字记录）**：仅有 `STUDENT_POLICY_CHANGELOG_ZH.md` 的叙述性描述，无可核实的原始运行文件；本报告中这类内容会明确标注“无独立量化文件”，不编造具体数字。

第 5 节起的每一张表都会在表注中标明对应等级。

---

## 5. 结果

### 5.1 总体结果（当前提交版本，公开集全量）

以下三组为公开集**全量 episode 一次性通过**的记录（数据来源：`right_policy_and_record/runs/task1_motion_debug/aggregate.json`、`task2_motion_debug/aggregate.json`、`task3_motion_debug/aggregate.json`）：

| 任务 | Episode 数 | Success rate | Target selection accuracy | Verification rate | Goal completion rate | 平均控制步数 | 平均耗时(s) | 危险接触总数 | 非目标物体最大位移(m) |
|---|---|---|---|---|---|---|---|---|---|
| Task1 | 6/6 | 1.00 | 1.00 | 1.00 | 1.00 | 440.0 / 800 | 106.25 | 0 | ≈1.8e-7（可忽略） |
| Task2 | 6/6 | 1.00 | 1.00 | 1.00 | 1.00 | 347.0 / 800 | 82.85 | 0 | 0.00142 |
| Task3 | 3/3 | 1.00 | 1.00 | 1.00 | 1.00 | 2286.0 / 3000 | 487.38 | 0 | 0.00000 |

*可信度：A 级（三份 `aggregate.json` 均直接读取）。*

三个任务在公开集上均达到 100% 成功率，控制步数与耗时均未超预算，全程无危险接触，非目标物体的意外位移可忽略。

> 说明：`right_policy_and_record/runs/` 下还保留了大量迭代过程中的失败与部分成功记录（如 `task2_fix_v19`～`v48` 多次 0% 通过、`task3_verify_v62`/`v65` 的 33% 通过），完整迭代轨迹见 `STUDENT_POLICY_CHANGELOG_ZH.md`；上表引用的是同一策略文件在迭代收敛后重跑的全量验证结果，而非选取历史中最好的单个 episode 拼接而成。

**全量危险接触核查**：对 `right_policy_and_record/runs/` 下全部约 138 份 `summary.json`逐一核查，`unsafe_contact_count` 字段均为 0，即从最早期失败版本到最终收敛版本，未出现过一次危险接触记录（危险接触判据与策略优劣无关，属于仿真物理层的硬约束，因此这一项在整个迭代历史上都是满足的，不代表早期版本已经“可用”）。

**retry_count 分布**（全部 138 份 summary.json 统计）：

| 分组 | 数量 | 占比 |
|---|---:|---:|
| success=true 且 retry_count=0 | 40 | 29.0% |
| success=true 且 retry_count>0 | 4 | 2.9% |
| success=false 且 retry_count=0 | 24 | 17.4% |
| success=false 且 retry_count>0 | 70 | 50.7% |

*可信度：A 级（对全部 summary.json 文件字段的直接统计）。* 可以看到多数失败 episode 并非“重试后仍失败”，而是重试机制被触发了很多次（>50%），说明验证/恢复逻辑在迭代过程中长期处于“频繁怀疑、反复重试”的状态，这也是 6.2 节把“验证与恢复”列为占比最大错误来源之一的数据依据。

### 5.2 按物体/语言细分（Task1 + Task2）

| Episode | 指令 | 物体→容器 | 控制步数 |
|---|---|---|---|
| t1_public_001 | 把红色方块放进方盘。 | red_cube → square_tray | 496 |
| t1_public_002 | Place the green cylinder into the round tray. | green_cylinder → round_tray | 457 |
| t1_public_003 | 请将蓝色长方体放入圆形盘子。 | blue_box → round_tray | 417 |
| t1_public_004 | Put the red cube in the round tray. | red_cube → round_tray | 406 |
| t1_public_005 | 把绿色圆柱体放到方盘里面。 | green_cylinder → square_tray | 371 |
| t1_public_006 | Place the blue rectangular block in the square tray. | blue_box → square_tray | 493 |
| t2_public_001 | 把香蕉放进圆盘。 | banana → round_tray | 352 |
| t2_public_002 | Place the apple into the square tray. | apple → square_tray | 357 |
| t2_public_003 | 请将黄色香蕉夹起并放入紫色圆盘。 | banana → round_tray | 328 |
| t2_public_004 | 抓取红苹果，放到黄色方盘中。 | apple → square_tray | 362 |
| t2_public_005 | Move the yellow banana to the round tray. | banana → round_tray | 357 |
| t2_public_006 | Pick up the apple and put it in the square tray. | apple → square_tray | 326 |

*可信度：A 级（步数直接取自各 episode 的 `summary.json`）。*

6 条指令全部成功，中英文、主动/被动句式之间控制步数差异不大（328–496 步），未观察到语言本身导致的系统性失败。

### 5.2b 逐 episode 异步控制细节（补充 5.1 的聚合平均值）

`async_control` 字段记录了每个 episode 的推理调度细节；以下按任务列出关键字段，用于说明“1.00 成功率”背后模型调用的实际负载而非仅看平均值：

| Episode | submitted | accepted | stale_dropped | model_errors | safe_hold_steps | reused_command_steps | last_latency_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| t1_public_001 | 218 | 218 | 0 | 0 | 0 | 278 | 0.29 |
| t1_public_004 | 179 | 179 | 0 | 0 | 0 | 227 | 0.30 |
| t2_public_001 | 155 | 155 | 0 | 0 | 0 | 197 | 0.31 |
| t2_public_004 | 160 | 159 | 1 | 0 | 3 | 202 | 0.32 |
| t3_public_001 | 1004 | 1003 | 1 | 0 | 12 | 1266 | 0.30 |
| t3_public_002 | 1171 | 1170 | 1 | 0 | 40 | 1231 | 0.31 |
| t3_public_003 | 947 | 946 | 1 | 0 | 22 | 1155 | 0.30 |

*可信度：A 级（数值抽样自各 episode 目录下 `summary.json.async_control`，非全部 15 个 episode 逐一列出，但每个任务均覆盖至少一个代表性样本；`stale_dropped`/`safe_hold_steps` 非零说明即便在全通过的最终版本中，异步调度仍会偶发丢弃过期指令并触发安全悬停，但不影响最终成功）。*

可以看到 `model_errors` 在这组最终通过的记录里均为 0，说明当前收敛版本下 VLM/检测服务调用是稳定的；但 `stale_dropped` 和 `safe_hold_steps` 不为零（尤其 Task3，因步数预算更长、单步耗时更久，触发次数也更多），说明命令复用/过期丢弃机制在正常运行中也会被触发，这是异步架构下的正常现象，而非异常。

### 5.3 按难度细分（Task3，多物体分拣）

以 `t3_public_002`（`right_policy_and_record/runs/task3_t002_verify_v66/t3_public_002/summary.json` 与 `task3_motion_debug/t3_public_002/summary.json` 两次独立通过记录）为例：

| 目标-容器配对 | 是否达标 |
|---|---|
| apple → round_tray | true |
| orange → round_tray | true |
| mustard_bottle → square_tray | true |
| potted_meat_can → square_tray | true |

4 组配对全部同时连续达标，`completed_goal_count=4`，`retry_count=0`，`unsafe_contact_count=0`。该 episode 在 v66 版本（`task3_t002_verify_v66`）中用 2638/3000 步完成，在后续 `task3_motion_debug` 重跑中用 2441/3000 步完成，两次运行步数存在约 200 步的正常波动（源自异步控制下的命令复用/延迟差异），但结果一致成功。

### 5.4 迭代过程指标（用于说明改进幅度，非最终结果）

| 阶段 | 代表版本/目录 | Success rate | Goal completion rate |
|---|---|---|---|
| Task3 早期 | `task3_final_v58` | 0.00 | 0.167 |
| Task3 几何防误配修复前的最后一次完整验证 | `task3_verify_v62` | 0.333 | 0.75 |
| Task3 张爪节奏+近处瓶子肩部抓取调整后 | `task3_verify_v65` | 0.333 | 0.917 |
| Task3 收敛（release_unjam 解卡逻辑加入后） | `task3_motion_debug` | **1.00** | **1.00** |

*可信度：A 级（四行均取自对应目录下的聚合/汇总文件）；但版本间的因果关系标注需要额外说明，见下方更正。*

> **更正说明（重要）**：changelog §13.5 记录的“几何防误配”修复是在 **v63** 引入的，而 `task3_verify_v62` 的运行数据产生于 v63 修复之前（changelog 原文明确写明“这不代表 v63 经过完整验收”，即 v62 是修复前最后一次完整验证，而非修复后的结果）。仓库中不存在独立的 `task3_*_v63*` 完整运行目录（`task3_unit_v63_tmp` 为空目录），因此 v63 修复本身**没有可核实的独立量化前后对比文件**，只能引用 changelog 的文字描述作为 C 级证据。上表中 `task3_verify_v62` 一行已改为准确描述“修复前最后一次完整验证”，不再误标为“修复后”。

`goal_completion_rate` 在整个迭代过程中持续上升（从 16.7% 到 100%），说明主要瓶颈从“完全无法完成任何子目标”逐步收敛为“个别子目标（尤其是瓶子放置后的斜卡误判）在最后一步失败”，最终在 v66 之后的版本中解决。

### 5.5 Task2 失败模式演化（v19→v50，迭代过程统计）

Task2 从 v19 到 v50 经历了约 30 个修复版本才达到全通过，期间失败记录可归为 5 类：

| 失败类别 | 代表 stage/现象 | 出现情况（迭代过程中） |
|---|---|---|
| `model_error` | VLM/检测服务调用失败或返回不可解析结果 | 约占该阶段失败记录的 45% |
| `align`/`transport` 超时 | 物体对齐或运输阶段耗尽单阶段步数预算 | 约占 20% |
| `safe_stop` | 触发安全停止（如检测到异常接触力/位置跳变）后未恢复 | 约占 15% |
| `recover`/`recover_view` | 视角丢失目标后触发恢复逻辑，但恢复未成功 | 2 个实例 |
| `retreat`/`verify_place` 近似失败 | 撤退或放置验证阶段边界判定，物体已基本到位但验证未通过 | 少数实例，属于判据过严导致的“伪失败” |

*可信度：B 级（基于对该区间内可用 summary.json/事件记录的 stage 字段分类统计，`task2_fix_v25/t2_public_001` 缺少 summary.json（运行被中断），故该 episode 未计入上述统计；由于并非每个版本都保留了全部 episode 的完整记录，上述占比为该阶段可用样本内的分布，不代表严格的全量统计）。*

这组分布与 6.2 节“错误来源定性占比”一致：`model_error` 与 `align/transport` 超时可归入感知/控制类问题，`safe_stop`/`recover`/`verify_place` 则更多是验证与恢复逻辑本身的判据问题——与 6.1 节 Task3 释放判据缺陷属于同一类根因（单边阈值判据在边界情况下不可靠）。

---

## 6. 失败分析

由于当前公开集三个任务均已达到全量成功，本节以**迭代过程中已定位并修复**的代表性失败链为例，说明诊断方法与错误归因方式，而非当前提交版本的未解决问题。

### 6.1 案例：Task3 第二项瓶子“斜卡误放”（v65→v66，同一 episode `t3_public_002` 对比）

以下时间线经**直接对 `events.jsonl` 做逐步 grep 核实**（而非仅转录 changelog 摘要），修复前（v65）与修复后（v66）为同一 episode（`t3_public_002`，seed=92）的两次独立运行：

**v65（release_unjam 逻辑加入前，最终失败）**：

| 控制步 | 阶段/观测 | 说明 |
|---:|---|---|
| ~1506 | `close`，开度异常偏高（≈1.03+） | 抓取几何导致瓶子斜卡在指间，抓取时即已埋下隐患 |
| ~1756 | `release` | 开爪后瓶子仍被撑住未脱离，但当时判据未能识别 |
| ~1812 | `retreat` | 原逻辑把“开度>0.97”单边当作释放成功判据，此处误判为已释放 |
| 3000（预算耗尽） | 瓶子目标 false，其余三件 true，`completed_goal_count=3` | v65 全程未进入任何解卡逻辑，最终该 episode 判定失败（4 个子目标只完成 3 个） |

**v66（release_unjam 状态引入后，最终成功）**：

| 控制步（events.jsonl 精确核实） | 阶段/观测 | 说明 |
|---:|---|---|
| **1831** | 进入 `release_unjam` | 检测到释放后开度仍异常，触发解卡状态（注：changelog 原文写作约“1812”附近，经 events.jsonl 逐条核对，实际进入点为 1831，两者相差约 20 步，属于 changelog 转录时的四舍五入误差，本报告以日志原始数据为准） |
| **1929** | 退出 `release_unjam`，进入 `retreat` | 原地小角度转腕解卡完成，开度回落到正常张开区间后才离开 |
| **1943** | 进入 `verify_place` | 解卡后位置验证，确认瓶子已正确留在方盘内 |
| 2638/3000 | episode 成功结束 | `completed_goal_count=4`，`retry_count=0`，`unsafe_contact_count=0` |

*可信度：B 级（步数 1831/1929/1943 为对 `events.jsonl` 原始事件直接核实所得，比 changelog 文字描述更精确；但由于历史版本目录并非每次都保留完整的 events.jsonl，此处对比基于可获得的两次独立运行记录，非严格控制其余变量的 A/B 实验）。*

**根因归类**：这是一个**控制/验证层面的判据缺陷**，不是感知或规划错误——检测与抓取目标本身是正确的（瓶子被正确识别、抓取动作也执行了），但“夹爪开度>0.97即视为释放完成”这一单边判据无法区分“正常张开”与“物理卡楔仍撑开夹爪”两种情况。修复方式（`release_unjam` 状态）：检测到释放后开度仍异常偏高时，先在容器上方原地小角度转腕解卡，等待开度真正回落到正常张开区间后才允许离开，而非直接机械执行“开→抬”。

### 6.1b 非目标物体异常位移案例（迭代过程中出现，最终版本未复现）

5.1 节报告的“非目标物体最大位移可忽略”是**最终收敛版本**的结果；但在迭代历史中曾出现两次数值异常的非目标位移，物理上不合常理（超出容器/桌面尺寸），记录如下，作为验证指标本身可能被极端异常值影响的例证：

| 版本/episode | 非目标物体 | 位移(m) | 说明 |
|---|---|---:|---|
| `task2_fix_v37/t2_public_001` | apple | 7.19 | 远超场景合理范围，推测为该物体被意外弹出场景或数值积分异常，而非正常的碰撞推挤 |
| `task2_fix_v39/t2_public_001` | apple | 3.35 | 同上，数量级同样不合理 |

*可信度：A 级（数值直接取自对应 summary.json 的 `non_target_displacement` 字段）。*

这两次异常均出现在 v37/v39 迭代阶段（早于最终收敛版本），且均为同一物体（apple）在同一 episode 编号下出现，提示当时的抓取/碰撞处理逻辑在特定物理条件下可能引发数值不稳定；由于最终收敛版本（`task2_motion_debug`）中该 episode 的非目标位移已恢复到正常量级（5.1 节表中 Task2 最大位移 0.00142m），且没有进一步的日志说明具体修复对应哪个版本变更，本报告只如实记录这一异常现象，不推断具体修复机制，避免过度解读。

### 6.2 错误来源占比（迭代过程整体归因）

依据 changelog 全文对各阶段问题的归类，并与 5.4 节 retry_count 分布、5.5 节 Task2 失败模式统计相互印证：

- **感知（检测/分割）相关**：约占早期问题的主要部分——托盘被误检为整张桌面（超大框）、YCB 物体轮廓检测不稳定、水果曲面质心偏差导致放置位置偏移。
- **规划（VLM/指令解析）相关**：占比较小——主要是 Task3 初期指令解析器只支持单物体格式，遇到分类指令即报错终止。
- **控制（状态机/IK/路径）相关**：占比最大——运输途中意外掉高度、容差冲突导致状态卡死、多段位移未做净空保护导致擦碰、远处瓶子竖直下探不可达。5.5 节统计的 Task2 `align`/`transport` 超时（约 20%）即属此类。
- **验证与恢复相关**：本节 6.1 案例即属此类——释放判据不完善导致的误判、水果摆动导致验证提前判失败、误将其他已完成物体当作待检索目标重新抓取。5.1 节统计的“success=false 且 retry_count>0”占全部记录的 50.7%，说明验证/恢复逻辑被触发的频率在整个迭代历史上都很高，是数据上最能佐证“此类占比最大”的一项。

*可信度：C 级为主（定性归因来自 changelog 叙述），感知/控制/验证三类的相对排序有 5.1/5.5 节的 A/B 级数据间接支撑，但没有对每一类错误做过精确的百分比统计，故仍以定性排序呈现，不给出具体百分比数字。*

### 6.3 已知数据缺口（如实说明，避免过度声称）

- `task2_fix_v25/t2_public_001`：运行被中断，无 `summary.json`，无法纳入该版本的统计。
- `task3_unit_v63_tmp`：空目录，v63（几何防误配修复）没有独立的量化前后对比文件。
- v9/v13/v14 等早期版本：changelog 中有文字变更记录，但未找到对应的独立运行目录，本报告不为这些版本编造具体数字。
- `sam_server.py` 的 DINO/OWLv2 检测级联：只有代码注释与 changelog 描述，没有落盘保存的检出率对比文件，无法给出量化消融结果（见 4.6 节 C 级说明）。

---

## 7. 讨论与边界

### 7.1 仿真与真实机器人的差异

- 本项目全程运行在 MuJoCo 物理仿真中，接触力、摩擦、抓取稳定性由仿真参数决定，与真实夹爪的柔性接触、传感器噪声、标定误差存在系统性差异；例如“夹爪开度>0.97”这类基于仿真理想执行器的判据，在真实硬件上可能需要额外的力/触觉反馈才能可靠区分“完全张开”与“物理卡楔”。
- 视觉输入为仿真渲染的 RGB-D，无真实摄像头的噪声、曝光、镜头畸变问题；检测模型（Grounding-DINO/SAM2/OWLv2）在真实场景下的表现可能与仿真渲染图像上的表现存在差距。

### 7.2 API/模型成本

- VLM 与检测服务均通过网络/本地 HTTP 调用，每次调用引入几百毫秒级延迟（实测 `last_latency_s≈0.31s`），命令复用机制（8 步窗口）用于摊薄这部分成本，但也意味着约一半的控制步是在“复用旧推理结果”而非“每步都重新决策”，这是一种在实时性约束下的必要折衷。
- 检测服务的级联策略（DINO 优先、OWLv2 兜底）在 DINO 失败时会额外触发一次推理，增加该次调用的总延迟，属于用延迟换检出率的取舍。

### 7.3 哪些结论不能推广

- 当前 100% 成功率是在**固定的 3 个/6 个公开 episode、固定 seed**上取得的，且每个 seed 只跑了一次，未做多次重复统计方差；不能直接推广为“该策略在任意随机场景下的成功率是 100%”，尤其是隐藏集/自建集若包含公开集未覆盖的物体摆放角度、光照条件或更极端的物体间距，可能暴露未被此次迭代覆盖的失败模式。
- 迭代过程中大量参数（高度阈值、容差、等待时长、倾角）是针对当前物理引擎版本、当前物体/容器几何尺寸手工调校得到的，更换物体模型或容器尺寸后很可能需要重新调参。

### 7.4 下一步最有价值的改进

- 补充多次重复运行（同一 seed 多次跑，或对同一任务生成更多随机 seed）以获得成功率的置信区间，而不是单次全量通过就宣称收敛。
- 检测服务侧的框筛选目前是启发式阈值（面积占比、置信度），可考虑引入更稳定的多视角一致性校验（如俯视+前视双相机检测结果的几何一致性核验），进一步降低误检风险。
- 状态机中大量启发式等待时长（settle/wait）可考虑替换为基于物理量（速度、加速度收敛）的自适应等待，减少不必要的时间开销，同时保留足够的稳定性判据。

---

## 8. 可复现性

### 8.1 环境

- 操作系统：开发过程中经历过 Windows + WSL2（Ubuntu 22.04）混合环境（涉及跨文件系统的权限位差异需注意，见 `git log` 中的相关清理记录），仿真与模型服务实际运行在 WSL2/Linux 侧。
- Python 版本约束：`pyproject.toml` 声明 `requires-python = ">=3.10,<3.14"`。
- 核心依赖（`requirements.txt`，精确锁定）：`mujoco==3.3.7`、`numpy>=1.24,<3`、`imageio>=2.34,<3`、`imageio-ffmpeg>=0.5,<1`、`pillow>=10,<12`、`typing-extensions>=4.6,<5`。
- 检测服务额外依赖（未在项目 requirements 中锁定版本，需以实际部署环境的 `pip freeze` 为准）：`torch`、`transformers`（`AutoModelForZeroShotObjectDetection`/`Owlv2ForObjectDetection`）、`sam2`、`Pillow`。
- 开发/测试依赖：`pytest>=8,<9`、`ruff>=0.6,<1`。

### 8.2 命令

启动检测服务（GPU）：
```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export HF_HUB_OFFLINE=1
python scripts/sam_server.py \
  --sam2-checkpoint ~/sam2-src/checkpoints/sam2.1_hiera_large.pt \
  --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --device cuda
```

运行评测（示例：Task1/Task2 全量，带视频）：
```bash
python -m graspbench.evaluate --policy policies.student_policy:StudentPolicy \
  --tasks configs/task1_place_public.json --output runs/task1_dev --video
python -m graspbench.evaluate --policy policies.student_policy:StudentPolicy \
  --tasks configs/task2_ycb_public.json --output runs/task2_dev --video
python -m graspbench.evaluate --policy policies.student_policy:StudentPolicy \
  --tasks configs/task3_sort_public.json --output runs/task3_dev --video
```

环境变量通过 `.env` 加载：
```bash
set -a; source .env; set +a
```
必需变量：`GRASPBENCH_VLM_BASE_URL`、`GRASPBENCH_VLM_API_KEY`、`GRASPBENCH_VLM_MODEL`、`GRASPBENCH_SAM3_URL`。

### 8.3 配置与版本

- 任务配置：`configs/task1_place_public.json`（seed 31–36）、`configs/task2_ycb_public.json`（seed 61–66）、`configs/task3_sort_public.json`（seed 91–93）。
- Prompt：VLM 与检测服务的提示词在代码中硬编码，未在本次迭代中修改（提示词文本本身视为固定协议，允许调整的是检测级联策略、框筛选阈值、几何后处理，而非提示词内容）。
- 策略文件版本标识：`policy_revision = "task123_sort_v66"`（内部版本号，记录在策略日志的 `debug.policy_revision` 字段，便于比对 `events.jsonl` 时确认加载的是哪一版代码）。
- 运行时间参考：Task1/2 单 episode 实测约 80–110 秒（wall time），Task3 单 episode 约 450–520 秒（wall time），主要受异步推理延迟与命令复用窗口影响，而非纯物理仿真耗时（Task3 的 `sim_time_s≈98s` vs `wall_time_s≈517s`，两者相差约 5 倍，差值主要来自等待模型服务响应）。

---

## 9. 团队分工

> 待补充：请提供参与本项目的团队成员名单及各自的具体分工（如：负责检测/感知模块修复、负责状态机与抓取控制、负责 Task3 分拣与解卡逻辑、负责实验记录与报告撰写等），以及哪些部分是共同完成的。当前初版报告的技术内容综合了 `right_policy_and_record/`（队友完成的完整可运行策略与实验记录，本报告的主要实验结果来源）与本人在 `policies/student_policy.py`/`policies/lean.py`/`scripts/sam_server.py`/`src/graspbench/perception.py` 等文件上的独立开发工作（检测误检修复、前倾抓取模块等），但具体的人员归属需要团队自行确认后补全本节。
