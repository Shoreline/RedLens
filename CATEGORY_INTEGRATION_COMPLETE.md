# Category Information 整合完成 ✅

## 修改内容

已将 Job Summary 的 **Statistics by Category** 和 **Detailed Entries** 整合在一起，按 category 分组显示。

### 文件修改

- **request.py**: 重构 `_generate_summary_html()` 函数中的 prebaked_html 生成逻辑

### 新的布局

每个 category 现在是一个独立的 section：

```
┌────────────────────────────────────────┐
│  Category Name            (蓝色标题栏) │
│  Total | Hits | Hit Rate | Attack Rate │
├────────────────────────────────────────┤
│  [图片1] CoMT ID, Backend, Path        │
│  [图片2] CoMT ID, Backend, Path        │
│  ...                                    │
└────────────────────────────────────────┘
```

### 优势

✅ 统计信息和详细图片整合在一起  
✅ 按 category 清晰分组  
✅ 一目了然，易于导航和对比  
✅ 不需要上下滚动对照  

### 已验证的 Job

- batch_5_0208_005316/job_159 ✅
- batch_5_0208_005316/job_160 ✅
- batch_5_0208_005316/job_161 ✅

### 效果预览

在浏览器中打开任一 job 的 summary.html 查看效果：

```bash
open output/batch_5_0208_005316/job_159_tasks_202_ComtVsp_qwen_qwen3-vl-235b-a22b-instruct_0208_005608/summary.html
```

---

**完成时间**: 2026-02-08  
**相关文档**: JOB_SUMMARY_CATEGORY_STATS.md
