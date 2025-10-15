global memory和L2 cache形成交互，L2通过mte2输入到L1 buffer, L0A buffer和L0B buffer.
L1 buffer通过mte1输入到L0A buffer, L0B buffer和BT buffer(bias table buffer)
L1 buffer通过FixPipe输入到FP buffer(fixpipe buffer)
L0A buffer,L0B buffer和BT buffer一起输入到Cube，得到L0C buffer.
FP buffer通过fix pipe连接了L0C Buffer和L1,L2 cache.

GetCoreMemSize接口可以获取不同类型ai处理器存储单元的大小

这是一个四级存储架构。
Global Memory最外层，容量最大，速度最慢。

Compute Cube最适合16*16的矩阵乘法。

所有通过搬运单元读写GM的数据都缺省被缓存在L2Cache，以此加快访问速度，提高访问效率。核外L2Cache以cacheline为单位加载数据，根据硬件规格不同，cacheline大小不同（128/256/512Byte等）。



共享文档内容，不一定可靠：
L2 cache的大小是192MB.
L1 buffer只在AIC中有，是512KB.
Unified buffer只在AIV中有，是192KB.
p个独立的L1 cache size = 524032 B \approx 512 KB
p个独立的L0A = L0B = 65536 B = 64 KB, L0C = 131072 B = 128 KB
