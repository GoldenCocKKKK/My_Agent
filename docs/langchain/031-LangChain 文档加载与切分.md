# LangChain 文档加载与切分

> 来源: https://www.runoob.com/langchain/langchain-document-loaders.html

---

# LangChain 文档加载与切分

上一章节我们介绍了 LangChain RAG，通过 Embedding 和向量数据库，让大模型能够根据外部知识回答问题。

但之前的示例主要使用手动编写的文本。在实际项目中，知识库可能来自 PDF、网页、Markdown、TXT、CSV，甚至整个文档目录。

要将这些内容用于 RAG，通常需要先经过两个步骤：Document Loader（文档加载） 和 Text Splitter（文本切分）。

本章节基于 LangChain 1.x，介绍常用文档加载器、Document 对象、中文文本切分、Markdown 标题切分以及完整的文档向量化流程。

---

## 一、为什么需要文档加载与切分？

一个典型的 RAG 知识库建立流程如下：

```
PDF / TXT / Markdown / 网页

              ↓

        Document Loader

              ↓

        Document 对象

              ↓

         Text Splitter

              ↓

        多个文档 Chunk

              ↓

        Embedding 模型

              ↓

          VectorStore

              ↓

        Retriever 检索
```

两个组件的职责不同：

| 组件 | 作用 | 输入 | 输出 |
| --- | --- | --- | --- |
| Document Loader | 读取并解析原始文档 | 文件、URL 等 | Document 列表 |
| Text Splitter | 将文档拆分成较小片段 | 文本或 Document | 文本片段或 Document 列表 |

文档切分的目的不只是满足模型长度限制，更重要的是提高知识检索的准确性。

例如，一份包含 100 个 Python 知识点的教程，如果整个文档只生成一个向量，搜索某个具体知识点时，可能受到大量无关内容的干扰。

将文档合理拆分后，检索系统就可以返回与问题更加相关的片段。

---

## 二、Document：LangChain 的统一文档格式

无论加载 TXT、PDF 还是网页，LangChain 通常都会将内容转换为 `Document` 对象。

Document 包含三个常用字段：

| 属性 | 说明 |
| --- | --- |
| `page_content` | 文档正文 |
| `metadata` | 来源、页码、标题等元数据 |
| `id` | 可选的文档标识 |

### 实例：手动创建 Document

## 实例

from langchain\_core.documents import Document

doc = Document(
    page\_content="Python 是一种通用编程语言，适合初学者学习。",
    metadata={
        "source": "python-tutorial.txt",
        "category": "python",
        "title": "Python 入门"
    }
)

print("内容：", doc.page\_content)
print("来源：", doc.metadata["source"])

输出：

```
内容： Python 是一种通用编程语言，适合初学者学习。

来源： python-tutorial.txt
```

`metadata` 非常重要，后续 RAG 检索可以利用它展示文档来源、过滤用户权限或定位原始资料。

---

## 三、常用 Document Loader

LangChain 提供多种文档加载器，覆盖常见的文件和网页内容。

| Loader | 支持格式 | 主要依赖 |
| --- | --- | --- |
| `TextLoader` | TXT、纯文本 | langchain-community |
| `PyPDFLoader` | PDF | langchain-community、pypdf |
| `WebBaseLoader` | 网页 HTML | langchain-community、beautifulsoup4 |
| `CSVLoader` | CSV | langchain-community |
| `UnstructuredMarkdownLoader` | Markdown | langchain-community、unstructured |
| `DirectoryLoader` | 目录内多个文件 | langchain-community 及对应 Loader 依赖 |

常用依赖安装：

```
pip install -U langchain-core langchain-community langchain-text-splitters pypdf beautifulsoup4 unstructured python-dotenv
```

如果需要构建向量知识库，可以继续安装：

```
pip install -U langchain-openai langchain-chroma
```

注意，不同 Loader 可能还有额外依赖。应根据实际文件格式安装，不需要一次性安装所有文档解析库。

---

## 四、TextLoader：加载 TXT 文件

`TextLoader` 用于读取本地纯文本文件，是最简单的文档加载器。

例如创建 `knowledge.txt`：

```
Python 是一种常用的编程语言。

Python 支持面向对象编程。

初学者可以先学习变量、列表、循环和函数。
```

### 实例：读取文本文件

## 实例

from langchain\_community.document\_loaders import TextLoader

loader = TextLoader("knowledge.txt", encoding="utf-8")
docs = loader.load()

print("文档数量：", len(docs))
for doc in docs:
    print("内容：", doc.page\_content)
    print("元数据：", doc.metadata)

对于普通 TXT，通常会返回一个 Document，其元数据包含源文件路径。

例如：

```
文档数量： 1

元数据： {'source': 'knowledge.txt'}
```

**注意：**`load()` 读取文件并返回 Document 列表，不代表文本已经完成切分。

对于较大的文本文件，还需要使用 Text Splitter。

---

## 五、PyPDFLoader：加载 PDF 文档

PDF 是企业知识库中常见的文档格式。

LangChain 可以使用 `PyPDFLoader` 提取 PDF 中的文本内容。

### 实例：读取 PDF

## 实例

from langchain\_community.document\_loaders import PyPDFLoader

loader = PyPDFLoader("python-guide.pdf")
docs = loader.load()

print("文档数量：", len(docs))

for doc in docs[:2]:
    print("页码：", doc.metadata.get("page"))
    print("来源：", doc.metadata.get("source"))
    print("内容：", doc.page\_content[:150])

通常情况下，按页加载 PDF 时，每页会生成一个 Document。

元数据可能包含：

```
{

    "source": "python-guide.pdf",

    "page": 0

}
```

其中页码通常从 0 开始。面向普通用户显示页码时，可以根据实际情况加 1。

### 扫描件 PDF 如何处理？

需要特别注意，PDF 不一定包含可提取的文字。

例如：

- 普通电子 PDF：通常可以直接提取文本。
- 扫描件 PDF：页面可能是图片，需要 OCR。
- 复杂表格 PDF：文本顺序可能与视觉布局不同。
- 双栏论文：可能需要进行阅读顺序和版面分析。

因此，对于合同、财务报表、试卷等需要高准确率的文件，不能只依赖 PDF 文本提取结果，应根据需要增加 OCR、版面分析与人工核对。

---

## 六、WebBaseLoader：加载网页

`WebBaseLoader` 可以读取网页并提取 HTML 中的文本。

例如，读取一个编程教程页面：

## 实例

from langchain\_community.document\_loaders import WebBaseLoader

url = "https://www.runoob.com/python/python-tutorial.html"
loader = WebBaseLoader(url)
docs = loader.load()

for doc in docs:
    print("来源：", doc.metadata.get("source"))
    print("内容：", doc.page\_content[:300])

需要注意，网页加载通常得到的是页面文本，不一定只有文章正文。

例如页面还可能包含：

- 顶部导航。
- 侧边栏。
- 底部版权信息。
- 相关推荐。
- 广告或其他非正文元素。

如果要将网页用于 RAG，建议进一步提取文章主内容，避免导航和重复页面元素影响检索。

**注意：**WebBaseLoader 不是完整浏览器。对于依赖 JavaScript 动态加载内容的网页，普通 HTTP 加载可能无法取得最终渲染后的文章。

批量抓取网站前，还应遵守目标网站的访问规则，并设置合理的请求频率。

---

## 七、CSVLoader：加载 CSV 数据

除了普通文档，LangChain 也可以加载 CSV 表格数据。

例如创建 `courses.csv`：

```
name,level,category

Python基础,入门,Python

Java基础,入门,Java

HTML基础,入门,前端
```

### 实例：读取 CSV

## 实例

from langchain\_community.document\_loaders import CSVLoader

loader = CSVLoader(file\_path="courses.csv", encoding="utf-8")
docs = loader.load()

print("记录数量：", len(docs))
for doc in docs:
    print(doc.page\_content)
    print(doc.metadata)

CSVLoader 通常会将每一行转换为一个 Document。

但要注意，CSV 中的大量数字、金额、统计指标，不一定适合全部转换为向量进行计算。

如果任务涉及精确筛选、求和、排序或统计，通常应该使用数据库、SQL 或 pandas 等结构化数据工具。

---

## 八、Markdown 文档加载

Markdown 常用于技术教程、开发文档和项目 README。

可以使用 `UnstructuredMarkdownLoader` 加载：

## 实例

from langchain\_community.document\_loaders import UnstructuredMarkdownLoader

loader = UnstructuredMarkdownLoader("README.md", mode="single")
docs = loader.load()

print("文档数：", len(docs))
print(docs[0].page\_content[:300])

这个示例会将 Markdown 解析为文本 Document。

对于希望保留标题层级的 Markdown 知识库，还可以结合 `MarkdownHeaderTextSplitter` 按标题切分。

这通常比单纯按照固定字符数拆分技术文档更有利于保留章节结构。

---

## 九、DirectoryLoader：批量加载整个目录

实际项目可能有几十甚至上百个文档。

可以使用 `DirectoryLoader` 按文件类型批量加载。

例如：

```
knowledge/

    ├── python.txt

    ├── java.txt

    └── html.txt
```

### 实例：批量读取 TXT

## 实例

from langchain\_community.document\_loaders import DirectoryLoader, TextLoader

loader = DirectoryLoader(
    "./knowledge",
    glob="\*\*/\*.txt",
    loader\_cls=TextLoader,
    loader\_kwargs={"encoding": "utf-8"},
    silent\_errors=False
)

docs = loader.load()
print("加载 Document 数量：", len(docs))

其中：

- `glob`：指定匹配的文件。
- `loader_cls`：指定实际使用的加载器。
- `loader_kwargs`：传入加载器参数。
- `silent_errors`：是否忽略部分加载错误。

如果目录中同时存在 PDF、TXT 和 Markdown，可以分别使用对应加载器读取，再合并 Document 列表。

---

## 十、Text Splitter：文档切分

文档加载完成后，下一步是将长文本拆分成适合检索的片段。

LangChain 将这些片段称为 Chunk。

例如：

```
一篇 3000 字的文章

         ↓

      TextSplitter

         ↓

Chunk 1：Python 基础语法

Chunk 2：变量与数据类型

Chunk 3：条件判断

Chunk 4：循环语句

Chunk 5：函数和模块

         ↓

      分别建立向量索引
```

理想情况下，每个 Chunk 应包含足够独立的知识内容，同时不要引入过多无关信息。

---

## 十一、RecursiveCharacterTextSplitter

`RecursiveCharacterTextSplitter` 是 LangChain 推荐的通用文本切分器。

它会按照一组分隔符递归尝试切分文本，尽可能保留段落、句子等自然结构。

对于中文内容，建议显式添加中文标点。

### 实例：切分中文教程

## 实例

from langchain\_text\_splitters import RecursiveCharacterTextSplitter

text = """Python 是一种简单易学的编程语言，适合初学者入门。

Python 支持多种数据类型，包括整数、浮点数、字符串、列表和字典。

条件判断用于控制程序流程，常见语法包括 if、elif 和 else。

循环可以重复执行代码，Python 提供 for 和 while 两种常见循环。

函数用于封装重复逻辑，可以接收参数并返回计算结果。"""

splitter = RecursiveCharacterTextSplitter(
    chunk\_size=65,
    chunk\_overlap=15,
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    length\_function=len
)

chunks = splitter.split\_text(text)

print("原文长度：", len(text))
print("片段数量：", len(chunks))

for i, chunk in enumerate(chunks, 1):
    print(f"\n--- Chunk {i}：{len(chunk)} 字符 ---")
    print(chunk)

这里故意将 `chunk_size` 设置为 65，方便观察切分效果。

运行后会打印实际生成的 Chunk 数量、内容和长度。

如果把 chunk\_size 改成 500，而原文长度不足 500，则通常只会得到一个 Chunk。

**注意：**切分结果取决于内容、分隔符、字符长度和重叠规则，不应把某次生成的 Chunk 数量当作固定结果。

---

## 十二、chunk\_size 与 chunk\_overlap

Text Splitter 最重要的两个参数是：

| 参数 | 含义 | 示例 |
| --- | --- | --- |
| `chunk_size` | 每个片段的目标最大长度 | 500 |
| `chunk_overlap` | 相邻片段的目标重叠长度 | 50 |

### chunk\_size

如果设置：

```
chunk_size=500
```

在默认的 `length_function=len` 下，长度通常按 Python 字符数计算，而不是 Token 数量。

中文字符通常按一个字符计数。

### chunk\_overlap

例如：

```
chunk_size=100

chunk_overlap=20
```

表示希望相邻片段之间保留一定的上下文重叠。

但 `chunk_overlap=20` 不表示每两个 Chunk 都一定恰好重叠 20 个字符。

实际重叠会受到段落边界和切分算法影响。

另外，重叠越多，重复索引的文本也越多，可能增加 Embedding 成本和检索结果重复率。

---

## 十三、切分参数设置指南

不同类型的文档，适合的切分方式可能不同。

以下是可以用于初步测试的字符数范围，并非固定最佳参数。

| 文档类型 | chunk\_size | chunk\_overlap | 建议 |
| --- | --- | --- | --- |
| FAQ 问答 | 200～500 | 0～50 | 尽量保持完整问答对 |
| 编程教程 | 500～1000 | 50～150 | 保留知识点与代码示例 |
| 技术文档 | 500～1200 | 50～150 | 优先保留章节结构 |
| 长文章 | 800～1500 | 100～200 | 避免上下文断裂 |
| 代码文档 | 按函数或类切分 | 按需设置 | 不要随意截断函数 |

例如，对于 Python 教程，建议尽量保持以下内容处于同一个片段：

```
知识点标题

    ↓

原理说明

    ↓

Python 示例代码

    ↓

运行结果与解释
```

如果代码与解释分散在不同 Chunk，检索到其中一个片段时，可能无法得到完整的知识内容。

---

## 十四、split\_text() 与 split\_documents()

LangChain Text Splitter 主要提供三种常用方法。

| 方法 | 输入 | 输出 |
| --- | --- | --- |
| `split_text()` | 一个字符串 | 字符串列表 |
| `create_documents()` | 字符串列表 | Document 列表 |
| `split_documents()` | Document 列表 | Document 列表 |

### 实例：保留来源信息

## 实例

from langchain\_core.documents import Document
from langchain\_text\_splitters import RecursiveCharacterTextSplitter

docs = [
    Document(
        page\_content="Python 是一种编程语言。" \* 100,
        metadata={"source": "python-guide.txt", "category": "python"}
    )
]

splitter = RecursiveCharacterTextSplitter(
    chunk\_size=200,
    chunk\_overlap=30,
    add\_start\_index=True
)

chunks = splitter.split\_documents(docs)

print("原始文档数：", len(docs))
print("切分后片段数：", len(chunks))

for chunk in chunks[:2]:
    print("来源：", chunk.metadata.get("source"))
    print("起始位置：", chunk.metadata.get("start\_index"))
    print("内容：", chunk.page\_content[:60])

使用 `split_documents()` 可以让切分后的 Document 继承原始文档的元数据。

`add_start_index=True` 还可以为片段记录起始字符位置，便于定位原文。

**建议：**在 RAG 项目中，优先使用 `split_documents()`，因为保留 source、page 等元数据非常重要。

---

## 十五、MarkdownHeaderTextSplitter：按照标题切分

对于教程、README 和技术文档，Markdown 往往具有清晰的标题层级。

例如：

```
# Python 教程

## 变量

## 数据类型

## 条件判断

## 循环语句
```

如果直接按固定长度切分，可能将不同章节混在一起。

可以使用 `MarkdownHeaderTextSplitter` 优先根据标题切分。

### 实例：保留 Markdown 标题信息

## 实例

from langchain\_text\_splitters import MarkdownHeaderTextSplitter

markdown = """# Python 教程
## 变量
变量用于保存数据。

## 列表
列表可以存储多个元素。

## 函数
函数用于封装重复逻辑。
"""

headers = [
    ("#", "章节"),
    ("##", "知识点")
]

splitter = MarkdownHeaderTextSplitter(headers\_to\_split\_on=headers)
docs = splitter.split\_text(markdown)

for doc in docs:
    print("内容：", doc.page\_content)
    print("元数据：", doc.metadata)

切分后，每个 Document 可以保留对应的标题层级。

例如：

```
内容： 变量用于保存数据。

元数据： {'章节': 'Python 教程', '知识点': '变量'}
```

这种方式特别适合按章节组织的编程教程。

如果某个 Markdown 章节本身仍然很长，还可以继续使用 RecursiveCharacterTextSplitter 进行二次切分。

---

## 十六、如何处理代码文档？

代码与普通文章不同，不能随意按照固定长度截断，否则可能把一个函数或类拆成不完整的两部分。

LangChain 的 `RecursiveCharacterTextSplitter` 支持根据编程语言设置分隔符。

### 实例：切分 Python 代码

## 实例

from langchain\_text\_splitters import RecursiveCharacterTextSplitter, Language

code = """
class Calculator:
    def add(self, a, b):
        return a + b

    def multiply(self, a, b):
        return a \* b

def hello(name):
    return f"Hello {name}"
"""

splitter = RecursiveCharacterTextSplitter.from\_language(
    language=Language.PYTHON,
    chunk\_size=100,
    chunk\_overlap=10
)

chunks = splitter.split\_text(code)

for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i}:")
    print(chunk)

这种方式会使用适合 Python 的分隔符，尽量在类或函数等结构附近切分。

但它并不等于完整的 Python AST 解析器，也不能保证所有函数都保持完整。

如果需要构建高质量代码库检索，可以进一步使用 Tree-sitter 或 AST 解析，在函数、类和模块边界处建立索引。

---

## 十七、完整流程：加载、切分、向量化

下面将前面的操作组合起来，建立一个简单的本地知识库。

流程：

```
本地 TXT 文件

     ↓

TextLoader

     ↓

Document

     ↓

RecursiveCharacterTextSplitter

     ↓

多个 Chunk

     ↓

OpenAI Embedding

     ↓

Chroma

     ↓

相似度检索
```

安装依赖：

```
pip install -U langchain-community langchain-text-splitters langchain-openai langchain-chroma python-dotenv
```

在 `.env` 中配置：

```
OPENAI_API_KEY=your_openai_api_key
```

### 完整实例：创建课程知识库

为了让读者直接运行，下面的程序会自动创建一份本地测试 TXT 文件。

文件名：`build_knowledge.py`

## 完整实例

import hashlib
from pathlib import Path
from dotenv import load\_dotenv
from langchain\_community.document\_loaders import TextLoader
from langchain\_text\_splitters import RecursiveCharacterTextSplitter
from langchain\_openai import OpenAIEmbeddings
from langchain\_chroma import Chroma

load\_dotenv()

# 1. 准备教学测试文档
path = Path("runoob\_knowledge.txt")
if not path.exists():
    path.write\_text("""Python 基础教程介绍变量、数据类型、条件判断、循环和函数。

Python 列表用于保存一组数据，可以通过索引访问元素。

Python 字典通过键值对组织数据，适合保存结构化信息。

HTML 教程介绍常用标签、链接、图片、表单及网页结构。

Java 教程介绍类、对象、继承、接口和异常处理。
""", encoding="utf-8")

# 2. 加载文档
loader = TextLoader(str(path), encoding="utf-8")
docs = loader.load()

# 3. 切分文档
splitter = RecursiveCharacterTextSplitter(
    chunk\_size=80,
    chunk\_overlap=10,
    separators=["\n\n", "\n", "。", "，", " ", ""],
    add\_start\_index=True
)
chunks = splitter.split\_documents(docs)

# 4. 初始化 Embedding
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 5. 连接持久化 Chroma
vector\_store = Chroma(
    collection\_name="runoob\_loader\_demo",
    embedding\_function=embeddings,
    persist\_directory="./runoob\_chroma\_db"
)

# 6. 使用稳定 ID 管理片段
ids = [
    hashlib.sha256(f"{path}:{doc.metadata.get('start\_index')}:{doc.page\_content}".encode()).hexdigest()
    for doc in chunks
]
vector\_store.add\_documents(chunks, ids=ids)

print("原始 Document 数量：", len(docs))
print("Chunk 数量：", len(chunks))

# 7. 语义检索
results = vector\_store.similarity\_search("Python 列表是什么？", k=2)

for i, doc in enumerate(results, 1):
    print(f"\n检索结果 {i}：")
    print("内容：", doc.page\_content)
    print("来源：", doc.metadata.get("source"))

运行后可能得到类似：

```
原始 Document 数量： 1

Chunk 数量： 5



检索结果 1：

内容： Python 列表用于保存一组数据，可以通过索引访问元素。

来源： runoob_knowledge.txt
```

实际 Chunk 数量和检索排序，取决于当前文档内容、切分参数与 Embedding 模型。

这里使用内容相关的固定 ID，避免同一份文档重复运行时不断生成新的随机 ID。

但如果原文发生修改，旧内容的 ID 可能仍留在数据库中。因此，生产环境还需要实现旧片段删除或基于文档版本的同步机制。

---

## 十八、索引建立与在线检索应该分开

RAG 系统通常分为两个独立流程：

```
build_index.py

    加载文档

    文档切分

    Embedding

    保存向量数据库



search.py

    连接向量数据库

    检索问题

    返回相关资料
```

建立索引通常只在文档新增或更新时执行。

用户每次提问时，不需要重新加载全部文档并生成 Embedding。

### 实例：连接已经存在的知识库

## 实例

from langchain\_openai import OpenAIEmbeddings
from langchain\_chroma import Chroma

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector\_store = Chroma(
    collection\_name="runoob\_loader\_demo",
    embedding\_function=embeddings,
    persist\_directory="./runoob\_chroma\_db"
)

results = vector\_store.similarity\_search("Python 如何入门？", k=3)

for doc in results:
    print(doc.page\_content)

这里使用与创建索引时相同的：

- Embedding 模型。
- Chroma 集合名称。
- 持久化目录。

**注意：**如果更换 Embedding 模型或向量维度，应该重新建立对应索引，而不是直接混用不同模型生成的向量。

---

## 十九、生产环境文档处理建议

### 1、保留文档来源

为每个 Chunk 保留 source、title、page、section 等元数据，方便后续展示来源和定位原文。

### 2、不要只依赖固定字符数切分

教程适合按章节切分，Markdown 适合按标题切分，代码适合按函数和类切分。

切分策略应与文档结构相匹配。

### 3、清理无关内容

网页中的导航、页脚和重复版权信息可能影响检索。

建议在建立索引前提取正文，并清理明显无关的页面元素。

### 4、确保表格和代码完整性

如果技术文档包含表格或代码示例，尽量避免将表头与数据、代码与解释完全分开。

### 5、处理扫描件 PDF

扫描件需要结合 OCR 才能获得可搜索文字。

对于精确性要求较高的合同和财务资料，应进一步验证文字识别结果。

### 6、实现索引更新和删除

文档修改后不仅需要加入新 Chunk，还应处理对应的旧向量，避免检索到已经过时的内容。

### 7、设置权限过滤

如果知识库包含用户私人文档，检索时必须根据可信用户身份进行权限过滤，不能把所有文档直接暴露给每个用户。

### 8、评估切分效果

可以选取真实用户问题，观察检索到的 Chunk 是否完整包含回答依据，再调整 chunk\_size、chunk\_overlap 和分隔符。

---

## 二十、常见问题

### 1、Document Loader 和 Text Splitter 有什么区别？

Loader 负责读取原始文档，Text Splitter 负责将文档拆分成小片段。

### 2、TextLoader 属于 langchain 核心包吗？

本教程使用的是 `langchain_community.document_loaders.TextLoader`，需要安装 langchain-community。

### 3、为什么设置 chunk\_size=500，只得到一个 Chunk？

如果原文本身不足 500 个字符，通常没有继续切分的必要。

### 4、chunk\_overlap=50 表示一定重叠 50 字符吗？

不是。它是目标重叠长度，实际结果受到分隔符和切分边界影响。

### 5、split\_text() 与 split\_documents() 应该选哪个？

普通字符串使用 split\_text()；需要保留来源元数据的 RAG 文档，通常优先使用 split\_documents()。

### 6、PDF Loader 能识别扫描件吗？

普通 PDF 文本提取不能保证识别扫描图片，需要根据文档情况加入 OCR。

### 7、Markdown 应该按字符还是标题切分？

有清晰标题结构的 Markdown 可以先按标题切分，再对过长章节进行二次切分。

### 8、代码文档适合使用普通文本切分吗？

简单项目可以使用语言感知的 RecursiveCharacterTextSplitter；复杂代码库建议使用语法解析器按函数和类切分。

### 9、为什么向量检索返回了不相关的片段？

可能与文档分块、Embedding 模型、检索数量、原始内容质量等因素有关。

### 10、文档修改后需要重新向量化吗？

受修改影响的文本片段通常需要更新 Embedding，并同步维护向量数据库中的旧片段。

### 11、加载网页时为什么有很多无关内容？

因为网页中还包含导航、页脚等元素，需要在索引前进行正文提取与清洗。

### 12、切分后是否需要重新保存来源信息？

使用 split\_documents() 通常可以继承原文档的 metadata，但仍建议检查实际结果是否保留所需字段。

---

## 总结

文档加载与切分是构建 LangChain RAG 知识库的重要基础。

本章节介绍了：

- `Document`：统一文档对象。
- `TextLoader`：读取 TXT 文件。
- `PyPDFLoader`：读取 PDF 文档。
- `WebBaseLoader`：读取网页。
- `CSVLoader`：读取 CSV 数据。
- `UnstructuredMarkdownLoader`：读取 Markdown。
- `DirectoryLoader`：批量加载目录。
- `RecursiveCharacterTextSplitter`：递归字符切分。
- `MarkdownHeaderTextSplitter`：按 Markdown 标题切分。
- `split_documents()`：保留元数据的文档切分。
- `Chroma`：将切分后的文档向量化存储。

最常见的处理流程：

```
loader = TextLoader("knowledge.txt", encoding="utf-8")

docs = loader.load()



splitter = RecursiveCharacterTextSplitter(

    chunk_size=800,

    chunk_overlap=100

)

chunks = splitter.split_documents(docs)



vector_store.add_documents(chunks)
```

> **核心原则：**RAG 的质量不只取决于大模型，文档加载是否准确、切分是否合理、来源信息是否完整，同样会影响最终检索与回答效果。
