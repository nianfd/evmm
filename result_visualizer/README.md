# Research Problem & Method Visualizer

这是一个纯前端交互式可视化工具，用来展示 `04_final_extraction.json` 的结果。

## 使用方式

直接用浏览器打开：

```text
result_visualizer/index.html
```

然后点击页面左上角的文件按钮，选择你的结果文件，例如：

```text
C:\project\papersmining\04_final_extraction.json
```

为了展开证据索引，建议继续加载同一次运行生成的两个中间文件：

```text
data\paper1\outputs\01_l1_chunk_results.json
data\paper1\outputs\02_evidence_index.json
```

如果你的最终文件来自 `C:\project\papersmining\04_final_extraction.json`，请找到对应运行目录下的这两个中间文件。加载后，右侧详情会把 `S003_C01:RP-1` 这类索引展开为：

- 所属 chunk 和章节
- L1 原始原子内容
- L1 原始 evidence
- 文本预览
- 相关图片名称

页面会展示：

- 研究问题节点
- 方法节点
- 问题-方法链接
- 证据覆盖、视觉证据、平均置信度等概览指标
- 按类型、粒度、置信度筛选
- 点击节点查看证据引用、风险说明和可复现字段

## 支持的 JSON 格式

推荐输入文件包含以下字段：

- `final_research_problems`
- `final_methods`
- `problem_method_links`
- `quality_report`

这正是当前工程输出的 `04_final_extraction.json` 格式。

## 证据索引含义

例如：

```text
S003_C01:RP-1
```

含义是：

- `S003_C01`：第 3 个 section 的第 1 个 chunk。
- `RP-1`：该 chunk 中 L1 阶段抽取出的第 1 个 research problem atom。

类似地：

```text
S006_C01:M-2
```

表示第 `S006_C01` 个 chunk 中的第 2 个 method atom。

## 连线类型

可视化中有两类边：

- 实线：`link_type = evidence_supported`，表示原流程抽取出的证据支持关系。
- 虚线：`link_type = inferred`，表示 relation completion 阶段补全的语义推断关系。

点击虚线边后，右侧详情会显示 `inferred`、置信度和补全理由。论文分析时建议把 inferred 边作为辅助展示，并进行人工复核。
