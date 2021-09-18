# RNN_POETRY

基于 RNN 生成写藏头诗，写绝句，随机写诗句

### 环境

- python3.6
- tensorflow 1.14.0

### 使用

- 训练：

<code>python poetry_gen.py --mode train</code>

- 随机生成：

<code>python poetry_gen.py --mode sample</code>

<code>巫峡红陵星别来，相公属望杳堪伤。盈浪临已解人去，今日江村弄云水。绿云虽出我真名，况值山川未死歌。但见西边断征客，又逢无事已千年。吴园肠断浦城北，八月曾归在路岐。</code>

- 生成藏头诗：

<code>python poetry_gen.py --mode sample --head 明月别枝惊鹊</code>

<code>明年能入五云前，月满铜台锁锦骝。别后信将诸我内，枝枝寥落日前枝。惊歌不驻歌繁过，鹊信春风出塞西。</code>

- 生成七言绝句：

<code>python poetry_gen.py --mode sample --head 我自横刀向天笑</code>

<code>我自横刀向天笑，有时苍虎皆久住</code>

- 生成五言绝句：

<code>python poetry_gen.py --mode sample --head 床前明月光</code>

<code>床前明月光，自恐奈归悲</code>

