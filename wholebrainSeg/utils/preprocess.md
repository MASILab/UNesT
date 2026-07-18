# 文件内容介绍：
/home/tenoke4090/B_WorkPath/mrqs/wholebrainseg_dataset/UIH164下为原始影像和标注结果，包含多个患者，如76384925062202为患者编号；

/home/tenoke4090/B_WorkPath/mrqs/wholebrainseg_dataset/UIH164/76384925062202/dcm目录下为原始DICOM影像；

/home/tenoke4090/B_WorkPath/mrqs/wholebrainseg_dataset/UIH164/76384925062202/c_results目录下，brain_preproc_img.nii.gz 为原始影像转到mini空间并区颅骨的影像；cleanup_labelmap96_src.nii.gz， 为brain_preproc_img.nii.gz对应的96空间下的标签结果


处理步骤：
1. 获取原始DICOM影像，并转成nii.gz格式
2. 获取原始dicom影像空间和brain_preproc_img.nii.gz空间之间的映射关系
3. 根据映射关系，将cleanup_labelmap96_src.nii.gz映射到原始影像空间


用户报告说程序在标签重采样过程中再次意外中断，显示"非法指令 (核心已转储)"错误。这表明之前添加的重试机制可能没有生效，或者问题出现在重试机制之外的地方。

让我先查看当前的代码，特别是重试机制的实现位置，看看是否需要调整。
程序在标签重采样时再次崩溃。让我检查当前的重试机制实现，看看问题出在哪里：
我需要查看 `_ants_resample_label` 方法的实现，看看重试机制是否正确实现：
问题在于"非法指令 (核心已转储)"错误是一个系统级别的崩溃（可能是SIGILL信号），这种错误不会被Python的try-except捕获，因为它是操作系统直接终止进程的。

这种错误通常是由以下原因导致：
1. CPU指令集不兼容（比如AVX/AVX2指令集问题）
2. 内存错误导致程序崩溃
3. ANTs或相关库的某些底层操作与当前CPU不兼容

解决方案：
1. 在调用 ants.apply_transforms 之前添加更积极的内存管理
2. 尝试设置环境变量来限制CPU指令集的使用
3. 考虑在子进程中执行重采样操作，这样崩溃不会终止主进程
4. 添加更低级别的处理，比如在调用ANTS前清理CUDA缓存

让我查看 `_ants_resample_label` 方法的完整实现，然后进行修改：
问题分析：**"非法指令 (核心已转储)"是系统级崩溃，Python的try-except无法捕获**。这通常由CPU指令集不兼容或内存问题导致。

让我查看完整的重采样方法并添加子进程隔离机制：
问题很明显：当前的重试机制只能捕获Python异常，无法处理系统级崩溃（如"非法指令"）。这类崩溃会直接终止整个进程。

解决方案：使用**子进程隔离**执行重采样操作。这样即使子进程崩溃，主进程也能检测到并重试。

我需要修改 `_ants_resample_label` 方法，将实际的ANTS重采样操作放入子进程中执行。
问题确认：系统级崩溃无法被Python异常捕获。需要使用**子进程隔离**机制：
文件内容可能已经变化，让我重新读取该部分。
现在我需要看到方法定义的开头，以便进行完整的替换：
-
现在我可以进行替换了，需要替换整个 `_ants_resample_label` 方法，添加子进程隔离机制。
修改成功。现在需要检查是否有语法错误，并确保import语句正确。让我检查文件开头的import部分，确保multiprocessing和其他必要的模块已导入。
-
lint检查通过，没有语法错误。现在让我看看是否有其他需要导入的模块，比如 multiprocessing、time、gc 等。让我继续读取import部分。
-
导入部分已经有了 `time` 和 `gc`，但需要添加 `multiprocessing`。不过由于我在方法内部导入 `multiprocessing as mp`，这是可以的。但为了更清晰，我可以在顶部添加导入。

实际上，在方法内部导入是更好的做法，因为 multiprocessing 在某些环境下可能有兼容性问题，而且只有在需要时才加载。

代码已经修改完成。让我给用户一个简洁的总结。
修改完成。主要改进：

**子进程隔离机制**：将ANTs重采样操作放入独立子进程执行，即使发生系统级崩溃（SIGILL、SIGSEGV等），主进程也能捕获并重试。

```
子进程退出码检测：
- exitcode=0: 成功
- exitcode=-4/-8: 非法指令 (SIGILL/SIGFPE)
- exitcode=-6: 异常终止 (SIGABRT)
- exitcode=-9: 被强制终止 (内存不足)
- exitcode=-11: 段错误 (SIGSEGV)
```

**新增特性**：
1. 使用 `multiprocessing` 的 `spawn` 模式创建子进程
2. 5分钟超时保护
3. 退出码解析，精准识别崩溃原因
4. 等待时间增加到 10/20/30 秒（指数退避）

现在可以重新运行脚本测试。