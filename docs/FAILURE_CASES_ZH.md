# 失败案例分析（综合我方与队友的实验记录）

版本：v1（初稿）
数据来源：我方 `runs/`（顶层，独立调试记录）+ 队友 `docs/runs/`（历史迭代记录）+ `right_policy_and_record/STUDENT_POLICY_CHANGELOG_ZH.md`（v1–v66 变更历史）+ `right_policy_and_record/run/`（最终收敛版本）

## 说明：可信度分级

本文档延续 `docs/EXPERIMENT_REPORT_ZH.md` 的证据分级方式：

- **A 级**：有 `summary.json`/`events.jsonl` 原始文件直接核实的数字（步数、位移量、goal_status、报错文本等）。
- **B 级**：结果数字可核实，但因果链（为什么会这样）缺少原始日志的直接证据，只能合理推断。
- **C 级**：仅有 changelog 的文字描述，没有找到对应的原始运行文件。

数据完整性说明：我方 `runs/` 顶层有 29 个子目录，其中 `inspect`/`manual_demo`/`pic`/`smoke`/`smoke_short`/`task3--video` 为调试快照或视频，非功能性失败案例，未纳入分析。队友 `docs/runs/` 约 80+ 个子目录，本文档重点核实了与 changelog 第 9–16 节直接对应的关键版本节点；`task3_unit_v*_tmp` 系列为空的单元测试临时目录，`pytest_v68_glfw` 为环境测试目录，均未纳入。

---

## 第一部分：我方失败案例

### 案例 1｜"清空视野"修复测试仍以超时收尾（`task1_clearview_test`）

- **证据**：`runs/task1_clearview_test/t1_public_001/summary.json`、`events.jsonl`
- **现象**：`last_stage="lift"`，`control_steps=800/800`（耗尽预算），`terminal_reason="max_steps"`，`red_cube` 位移仅 0.0071m。
- **分析**：目录名暗示这是在测试"抓取前移开手臂清空相机视野"相关的修复，但从结果看，机械臂卡在 `lift`（抬升）阶段直到超时，物体几乎没有被搬运。说明这次修复没有解决抓取后卡顿的问题。
- **可信度**：A 级

### 案例 2｜对照组"不清空视野"：更早触发 model_error（`task1_no_clearview_test`）

- **证据**：`runs/task1_no_clearview_test/t1_public_001/summary.json`
- **现象**：`control_steps=28`，`last_stage="model_error"`，`selected_target=null`。
- **分析**：VLM 服务对该指令（红色方块，seed 31）返回了无法解析的响应，触发 model_error 提前终止，比案例1更早失败（28步 vs 800步）。这更可能是同一服务间歇性问题的偶然复现，而非"关闭清空视野逻辑导致更差结果"的因果关系，报告中应表述为"未见改善"而非"变差"。
- **可信度**：A 级

### 案例 3｜"框体过滤"修复后仍 4 次重试放置失败（`task1_box_filter_test`）

- **证据**：`runs/task1_box_filter_test/t1_public_001/summary.json`
- **现象**：`control_steps=744`，`last_stage="verify_failed"`，`retry_count=4`，`red_cube` 位移仅 0.003m。
- **分析**：目标选择正确（`target_selection_correct=true`），但连续 4 次放置验证都失败，物体几乎没有移动——暗示抓取本身可能没有真正抓到物体（一直在"假抓"），而不是放置位置计算错误。这与队友版本中"误重抓"类问题（见共性问题C）性质类似。
- **可信度**：A 级

### 案例 4｜蓝色长方体指令稳定触发 VLM 任务规划 JSON 解析失败

- **证据**：`runs/task1_align_test/t1_public_003/events.jsonl`（第30步）、`runs/task1_square_align_test/t1_public_003/events.jsonl`（第31/32步）
- **精确报错文本**：`"Model service failed during planning; holding safely: Invalid VLM task-plan response (expected actions and reason JSON)"`
- **现象**：三次运行均在约30步左右终止，`selected_target=null`。
- **分析**：VLM 返回的任务规划响应不是预期的 `actions`+`reason` JSON 结构，策略只能安全停止。这是服务端/模型输出格式不稳定导致的，不是坐标计算错误，且在两个不同调试版本中复现，说明问题具有一定持续性或随机性。
- **可信度**：A 级（两份独立 events.jsonl 交叉印证同一报错文本）

### 案例 5｜任务已达成但被末尾 model_error 拖累为失败（`task1_restore_test1`）

- **证据**：`runs/task1_restore_test1/t1_public_001/summary.json`、`events.jsonl`（第757–796步）
- **阶段序列**：`...→release(728)→rise_after_place(752)→async_model_error(796)`
- **关键发现**：`goal_status: {"red_cube->square_tray": true}`（目标已physically达成），但 `success=false`，`last_stage="async_model_error"`。第796步的 model_error 发生在收尾抬升阶段，此时任务已经完成。
- **分析**：即使物体已经正确放置，最后一次异步模型调用失败也会让 episode 以非 success 状态收场。这是"任务完成度"与"episode 最终判定"可能脱节的典型案例，值得在报告中单独说明。位移量0.401m经核实主要来自释放后的正常物理弹跳（descend_place 阶段末端误差已降到0.009m，释放精度本身没有问题）。
- **可信度**：A 级

### 案例 6｜高位移不代表失败：正面对照（`task1_shift_start_test`）

- **证据**：`runs/task1_shift_start_test/t1_public_001/summary.json`
- **现象**：`success=true`，但 `red_cube` 位移达 0.395m。
- **分析**：与案例5一起验证了"位移量本身不是判断成功/失败的可靠指标"——方形物体释放后的弹跳位移即使较大，只要没有滚出容器就不影响成功判定。报告中应提示：应结合 `goal_status` 而非单纯位移数值判断结果。
- **可信度**：A 级

### 案例 7｜Task3 三次尝试全部因目标选择/规划错误失败（`task3_lean_test`）

- **证据**：`runs/task3_lean_test/t3_public_{001,002,003}/summary.json`
- **现象**：三项均 `control_steps=3000/3000`（耗尽预算），`completed_goal_count=0`（全部0/4）；`t3_public_001` 中 `selected_target="mustard_bottle"` 但 `expected_target="apple"`。
- **分析**：目录名 "lean" 暗示这是在测试倾斜抓取姿态（应对瓶子难以垂直抓取的问题），但从数据看，问题出在目标选择/任务规划阶段就没有配对正确，距离真正验证"倾斜抓取"效果还差一步——这次测试的前置条件本身不成立，倾斜抓取能力未被验证到。
- **可信度**：A 级

### 案例 8｜策略自认为完成但评测判定失败：短时验证与最终稳定态的缝隙（`task2`）

- **证据**：`runs/task2/t2_public_001/summary.json`（banana，位移0.184m）、`t2_public_004`（apple，位移0.320m）、`t2_public_006`（apple，位移0.465m）——三项 `goal_status` 均为 `false`
- **关键日志**（`t2_public_004` 第766–768步）：`"Re-detected the target at its expected location; this pick/place is complete."` → `"All planned pick/place actions finished."`，`done=true`
- **分析**：策略自身的 `verify_place` 判定基于重新检测目标位置一次通过就宣布完成，但评测器最终判定为失败。释放动作本身没有明显问题（descend_place 末端误差已降到0.001m），随后物体位移（0.32~0.47m）说明苹果/香蕉释放后发生了滚动，而策略的"重新检测"发生在滚出容器之前的短暂窗口内，因此策略自认为成功但物体后续滚出了托盘。**这是本文档中最值得关注的案例之一**：暴露了"基于单帧/短时重检测的自我验证机制"与"评测器要求的最终稳定状态"之间的系统性缝隙——这与队友版本中同类问题（见共性问题A）完全对应。
- **可信度**：A 级（3个独立 episode 相同模式）

---

## 第二部分：队友失败案例（已用原始日志逐步核实）

### 案例 9｜Task3 瓶子"斜卡误放"（release_unjam 修复前后对比）

- **证据**：修复前 `docs/runs/task3_verify_v65/t3_public_002/summary.json`；修复后 `docs/runs/task3_t002_verify_v66/t3_public_002/summary.json`、`events.jsonl`
- **结果对比**：v65 `success=false`（3000步耗尽，4项中瓶子`mustard_bottle`未达标，其余3项达标）；v66 `success=true`（2638/3000步，4/4全部达标）。
- **精确时间线**（events.jsonl 逐步核实，第三次抓放循环即瓶子）：
  - 第1775步进入 `release`
  - 第1828–1830步：`gripper_opening`≈1.0165（超过1.0的"正常张开上限"，说明夹爪被瓶子撑开卡住）
  - **第1831步**：进入 `release_unjam`（比 changelog 转述的"约1812"更精确20步）
  - 第1925–1928步：开度逐渐回落至约1.001–1.003
  - **第1929步**：开度=1.0006，进入 `retreat`
  - **第1943步**：进入 `verify_place`
- **根因**：归一化夹爪开度命令上限是1，但物理关节在物体卡住时实际开度可超过1；旧判据"开度≥0.97即认为释放成功"是单边阈值，无法区分"正常张开"与"被斜卡物体撑开"，导致瓶子被误判为已释放。
- **修复方式**：新增 `release_unjam` 阶段——检测到释放后开度仍>1.005，原地低速转腕解卡，等待开度回落到0.97~1.003区间且关节速度<0.10才允许撤离。
- **可信度**：A 级（关键数字与changelog精确吻合，无四舍五入偏差）

### 案例 10｜Task2 非目标物体异常大位移（apple 7.19m / 3.35m）

- **证据**：`docs/runs/task2_fix_v37/t2_public_001/summary.json`（`banana`位移0.243m，`apple`位移**7.192m**）；`docs/runs/task2_fix_v39/t2_public_001/summary.json`（`apple`位移**3.349m**）
- **阶段序列**（v37 t2_public_001）：`align→approach→close→lift_clear_1→verify→transport→recover→recover_view→recovery_settle→retry_center→pregrasp→...`（循环三轮，最终仍以800步耗尽、未成功放置banana）
- **分析**：`events.jsonl` 不包含物体真值位置（只有机器人关节/末端/相机观测），无法从原始日志精确定位苹果7.19m位移发生在哪一步。合理推测：banana多次"抓取失败→recover→retry_center"过程中机械臂在桌面附近大幅摆动，可能碰撞到旁边的apple将其扫飞；但这只是推测，`unsafe_contact_count`字段仍为0，说明系统内建的危险接触计数器没有捕捉到这次撞击——这本身提示碰撞检测机制可能存在盲区，值得在报告中提及。
- **可信度**：B 级（位移数值本身A级，但"机械臂碰撞导致位移"这一因果链无法直接核实）

### 案例 11｜Task2 从 v19 到 v50 的失败演化（27个版本、45+ episode）

- **证据**：`docs/runs/task2_fix_v{19,20,20b,20c,21,23,24,26,27,28,29,30,31,32,33,34,36,37,38,39,41,42,43,44,47,48,49,50}/*/summary.json`
- **分布统计**：
  - **model_error 主导**（v19–v39区间高频出现）：19个版本出现 `last_stage="model_error"`/`"async_model_error"`，是这一长阶段的主要瓶颈。
  - **align 超时**：v19,20,20b,23,26,29,32,36,38（对齐阶段反复卡死耗尽800步预算）。
  - **safe_stop**：v31,34,41,42,43,47（策略主动安全停止）。
  - **verify_place 首次出现于 v48**（验证逻辑首次跑通，但判定仍未通过）。
  - **v49 首次全部成功**，**v50 六项全部成功**（`aggregate.json`: `success_rate=1.0`，`mean_control_steps=349.33`，`total_unsafe_contacts=0`）。
- **分析**：这是一条清晰的收敛曲线——从"服务调用不稳定为主"到"对齐反复超时"到"安全停止"再到"验证逻辑跑通"最终到"全部成功"，反映的是渐进式调试过程，而非单次修复。
- **可信度**：A 级（45个summary.json逐一读取核实）

### 案例 12｜几何防误配问题（v62 修复前的最后完整验证）

- **证据**：`docs/runs/task3_verify_v62/t3_public_{001,002,003}/summary.json`
- **结果**：`t3_public_001` 成功（4/4）；`t3_public_002` 失败（2/4，apple、orange未达标）；`t3_public_003` 失败（3/4，mustard_bottle未达标）。
- **根因（changelog 文字，events.jsonl 中无法找到直接证据）**：遮挡导致水果 mask 点云拟合半径过大，可能将未抓取的另一颗水果误判为已放置的目标。
- **重要说明**：changelog 明确指出 v63（几何防误配修复本身）没有独立量化文件（本次核查确认 `docs/runs/` 中确实没有对应的完整验证目录），因此 v63 的修复效果无法用原始文件直接核实，只能以 v62（修复前）的失败与后续版本（v64/v65/v66）的部分/完全成功作间接佐证。
- **可信度**：B 级（失败结果数字A级，具体几何根因机制为C级）

### 案例 13｜Task1 抬升容差与托盘定位偏差的收敛过程（v12→v18）

- **证据**：`docs/runs/task1_v12/t1_public_{001..006}/summary.json`（6项全部失败，`last_stage="lift_inward"`，800步耗尽）；`task1_t003_fix_v16`/`v17`（`success=false`，`retry_count=2`）；`task1_t003_fix_v18`（`success=true`，417步，`retry_count=0`）
- **根因**：v12阶段抬升容差过严（6mm），带载残差10.7/9.6mm超时触发误开爪；v16/v17阶段托盘中心定位偏差约2cm导致反复撞盘沿/浅夹持滑脱。
- **修复方式**：v18 重新估计托盘点云边界中点，并抬高运输/松爪高度。
- **可信度**：A 级（9个summary.json逐一核实，数字精确匹配changelog）

### 案例 14｜模型服务未启动导致 Task3 提前 safe_stop

- **证据**：`docs/runs/task3_fix_v51_service_check/t3_public_001/summary.json`、`events.jsonl`
- **精确报错**（第4步）：`"Stopping safely: ModelServiceError: Model service http://127.0.0.1:8765/infer failed: <urlopen error [WinError 10061] 由于目标计算机积极拒绝，无法连接。>"`
- **现象**：仅5步即触发 `safe_stop`。
- **分析**：本地模型服务未启动/连接被拒绝，属于环境配置问题而非策略逻辑问题。
- **可信度**：A 级

---

## 第三部分：跨人共性问题

### 共性问题 A｜"策略自我验证通过" ≠ "评测器最终判定通过"

- **我方证据**：案例8（`verify_place`/`complete` 阶段判定完成，但 `goal_status=false`）。
- **队友证据**：changelog 第13.1节"取消提前终止"（v59，"单帧条件成立就直接done=True，可能截断评测所需的连续稳定窗口"）；案例9（"开度≥0.97"单边判据被误接受）；`task3_verify_v65`（末帧4/4却因需连续多帧保持而判false）。
- **结论**：这是双方独立发现的同一类系统性问题——基于单帧/短时重检测或单一阈值的自验证机制，与评测器要求的最终物理稳定状态之间存在结构性缝隙。建议作为报告的重点论述对象。

### 共性问题 B｜圆形/曲面物体释放后易滚动移位

- **我方证据**：案例8（苹果/香蕉位移0.32~0.47m滚出托盘）；案例6（方块位移0.39m仍成功，因位移是原地弹跳而非滚出）。
- **队友证据**：changelog第13.4节专门处理苹果/橙子球心估计偏差；第17节"单纯延长松爪前等待，苹果结果不单调"；`task3_verify_v65`"苹果摇摆使四项未连续保持10帧"。
- **结论**：双方都独立发现"球形/曲面物体（苹果、橙子）比方形物体更难稳定放置"，是共性物理难点，非各自代码的偶然bug。

### 共性问题 C｜检测框/分割框过大导致误检测

- **我方证据**：`task1_box_filter_test`（案例3，目录名直接暗示测试"过滤过大检测框"，部分改善但仍4次重试失败）。
- **队友证据**：changelog第7.1节（v9）"方盘mask占全图约54.8%，box占74%，实际选中了大半场景"，修复为"拒绝mask占比大于0.22、box占比大于0.30的结果"。
- **结论**：双方独立遇到并各自修复了"分割/检测框过大误判为托盘或物体"的问题，是 VLM+SAM 视觉管线的共性缺陷，值得对比两种解法。

### 共性问题 D｜VLM/模型服务偶发性调用失败（model_error）贯穿始终

- **我方证据**：案例4（VLM JSON解析失败）、案例2/5（model_error导致episode提前结束或干扰终值判定）。
- **队友证据**：案例11（v19–v50期间近20个版本反复出现model_error为主导失败阶段）、案例14（服务未启动的safe_stop）。
- **结论**：model_error 是双方项目中贯穿始终、频率最高的失败类别，不完全是策略代码问题，很大比例源于外部模型服务的不稳定性/环境配置问题。报告中应明确区分"策略逻辑bug"与"外部服务不稳定"两类失败根因，避免混淆。

---

## 数据完整性说明

- 所有标注 A 级的案例均已直接核对对应 `summary.json`/`events.jsonl` 原始文件，未发现与 changelog 描述不吻合的情况。
- 案例10（apple 7.19m/3.35m位移的根因）是本次核查中唯一"结果数字可信但因果链不可直接核实"的情况，已明确标注为B级。
- 案例12（几何防误配 v63）的具体几何根因机制为C级，因为没有独立的量化前后对比文件，仅有 changelog 文字描述。
