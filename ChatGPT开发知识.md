## ChatGPT开发知识

[TOC]

### 第二期：用ChatGPT构建对话系统

1. 当让ChatGPT处理字符型任务时，直接输入单词来处理其中的字符会比较困难。

   这是因为英文的提词器（tokenizer）并非以单词为token，而是以几个字符的合并作为token。

   在单词的字符中增加 `-` （英文连字符）符号，可以增加字符处理成功的概率。

2. 对于英语来说，一个token大概是4个字符，或者是一个单词的四分之三。

   不同的大语言模型会对输入输出的token数量进行限制。

   输入文本通常被称为“上下文”，输出通常被称为“完成结果”。

   GPT-3.5的输入和输出token数量大概是4000个，超过4000个token很容易发生异常。

   这么换算下来，实际文本的总长度不应该超过16000的英文字符~

3. 聊天机器人的编写格式。

   `system` 这个键里面的文本告诉ChatGPT生成文本的风格、语气等等，为聊天机器人奠定基调；

   `user` 这个键里面的文本则是用户当下对已经具有一定语言风格的ChatGPT提出具体实际的需求。

   要让ChatGPT知道之前说过了什么。

4. 如何知道自己使用了多少token？

5. 如何在使用ChatGPT的openai-key时候不会被泄露？

   使用 `dotenv` 这个库，将自己的openai-key写进隐藏文件 `.env` 中，然后通过代码实现隐式地调用openai-key。具体代码如下。

   ```python
   from dotenv import load_dotenv, find_dotenv
   _ = load_dotenv(find_dotenv())
   openai.api_key = os.getenv("OPENAI_API_KEY")
   ```

6. 对于需要处理不同情况下的大量独立指令集任务，首先对查询任务进行分类，然后利用分类结果确定要使用哪些指令可能会很有益处。可以通过固定类别和硬编码与处理类别任务相关的指令来实现。

7. 在prompt提示词中，要用分割符分割不同内容，也就是让ChatGPT看到不同的内容，就像中学生做题目一样，并非文案里面的所有文字都是问题，所以要用分隔符隔开一下。视频里建议采用 `####` 来分割，从上文可知，这就是一个token，可以很好地分割文本。

8. Moderation API 指的是在确保内容符合 OpenAI 的使用政策，这些政策反映了我们能安全、负责任地使用AI技术。Moderation API 能帮助开发者识别和过滤各种禁止的内容，比如仇恨、自残和性暴力。他还有子类别，实现更加精确的内容分类。

9. 什么是“提示注入”？

   用户试图通过提供输入来操纵AI系统，以试图覆盖或者绕过您作为开发人员设置的预期指令和约束条件。

   “提示注入”会让AI系统进入不合理的工作状态，带来效益和成本的损失，比如说，你用客服机器人帮你写英文小作文，其实是在损耗客服机器人商家的成本hhhhhh。

10. 如何避免“提示注入”？

    - 使用分隔符分隔内容和引导性内容。分隔符尽量不被知道，最好是4个字符左右。
    - 在用户输入文本前，先用字符串替换函数，将里面可疑的分隔符换成我们的分隔符，避免用户得到这个系统的分隔符后对系统造成干扰。
    - 在构建系统前，先用ChatGPT分析出这是不是“提示注入”，让ChatGPT自己回答并影响后续步骤。

11. 用户模型可能会在匆忙得出错误结论时发生推理错误，因此我们可以重构查询，要求模型在提供最终答案之前给出一系列相关的推理步骤 ，这样它可以更长时间、更有条理地思考问题。

    通常情况下，我们称之为要求模型按步骤推理问题的策略为思维链推理。

    在实际应用中，模型的思维链推理一般不给用户知道。可以采用一种“内心独白”的方法进行隐式地推理。内心独白的思路是指示模型将应该对用户隐藏的部分输出放入结构化的格式中，以便于传递。在向用户呈现输出之前，将输出经过处理，只显示部分内容。

12. 总的来说，确定何时动态加载信息到模型的上下文中，并允许模型在需要更多信息时决定，是增强这些模型能力的最佳方式之一。应该将语言模型视为一个需要必要上下文才能得出有用结论和执行有用任务的推理代理。模型实际上很擅长决定何时使用各种不同的工具，并能够按照指示正确使用它们。

13. 如何评估ChatGPT生成的输出？

    - 第一种方式是使用 Moderation API 正如上文提到的；

    - 第二种方式是让ChatGPT生成的输出作为输入的文本，设置合适的提示词和分隔符，将输出内容作为输入，让ChatGPT自己来评判自己生成的文本是否合理。

      在这类评估任务中，使用更高级的模型（gpt-4）更好，因为它们在推理方面更出色。

      针对第二种方法， `Does the response use the retrieved information correctly?` 和 `Does the response sufficiently answer the question?` 是两个比较好的prompts。

      很少人采用第二种方式进行生成内容的评估，这是因为徒增token的数量（token是要花钱的......

---

### 第三期：用大语言模型的LangChain开发应用

#### 基本介绍

1. 什么是LangChain?

   为==大语言模型==构建的==开源框架==；

   具有==Python==和Java等两个packages；

   特点是专注于==组件==和==模块化==，主要体现在：1）模块化的组件，可以调库导包的形式使用；2）有很多的用户组件可以==帮助构建LLM开发==；

2. LangChain主要实现了什么？

   传统的LLM不能联网找到答案，不能输入用户自己的数据进行回答。

   因此大佬设计了LangChain，在我看来主要是实现了==超LLM数据集外==的内容的生成、问答等等。

3. LangChain中文教程：https://liaokong.gitbook.io/llm-kai-fa-jiao-cheng/

---

#### 内容

1. 什么是 `Loader` 加载器？

   这个就是==从指定源中加载数据的工具==。算是一种获得外源信息的途径。

   - 文件夹：`DirectoryLoader`
   - Azure 存储：`AzureBlobStorageContainerLoader`
   - CSV文件：`CSVLoader`
   - 印象笔记：`EverNoteLoader`
   - Google网盘：`GoogleDriveLoader`
   - 任意的网页：`UnstructuredHTMLLoader`
   - PDF：`PyPDFLoader`
   - S3：`S3DirectoryLoader / S3FileLoader`
   - Youtube：`YoutubeLoader`

2. 什么是 `Document` 文档？

   当使用 `loader` 加载器读取到数据源后，数据源需要==转换成Document对象==后，后续才能进行使用。

3. 什么是 `Text Spltters` 文本分割？

   文本分割的作用顾名思义就是分割文本的，之所以需要分割文本，是因为当==外源信息很大==（300页的pdf文件）时，直接输入到ChatGPT里面会报错，因此需要分割以满足ChatGPT==最大token的限制==。

4. 什么是 `Vectorstores` 向量数据库？
   向量数据库是存储向量数据的，==数据在数据库中的检索实际是一种向量计算==。

   因此要把分割完的 `Document` 对象转换成向量数据库，便于后续的检索。

5. 什么是 `Chain` 链？

   教程作者认为，一个==Chain链就是一个文本任务==（例如：摘要、翻译和生成文字等）。

   一个个Chain链可以通过一些逻辑关系连起来，就好像一个链条一样。

6. 什么是 `Agent` ？

   简单的理解为：可以动态的帮我们==选择和调用chain==或者==已有的工具==

   类比于强化学习智能体，根据当前的状态（在LangChain里面就是文本）执行不同的动作。

   在LangChain里面，针对不同的外源数据采用不同的“动作”（也就是一些内置的、自定义的工具）。

7. 什么是 `Embedding` ？

   用于衡量文本的相关性。

   相比 fine-tuning 最大的优势就是，不用加一次新的内容就训练一次，成本要比 fine-tuning 低很多。

   > 教程在前面说了“文本相关性”，后面又说可以优化 fine-tuning 过程。
   >
   > 没学过LLM的具体内容不太理解额。


---

#### 项目1. 使用LangChain完成一次问答

```python
import os
from langchain.llms import OpenAI

os.environ["OPENAI_API_KEY"] = 'sk-your-OpenAI-key'

LLM = OpenAI(model_name='text-davinci-003',
             max_tokens=2048)

r = LLM("Please tell me a joke.")

print(r)
```

#### 项目2. 使用LangChain实现LLM通过Google搜索获得资料。

- 进行Google搜索需要使用Google提供的API接口。

  教程如下：首先需要我们到 Serpapi 官网上注册一个用户，https://serpapi.com/ 并复制他给我们生成 api key。

- 一些主要库和对应版本列表如下：

  google-search-results==2.4.2

  langchain==0.0.200

  langchainplus-sdk==0.0.10

  openai==0.27.8

  openapi-schema-pydantic==1.2.4

- 代码实现如下

  ```python
  import os
  from langchain.agents import initialize_agent
  from langchain.agents import load_tools
  from langchain.agents import AgentType
  from langchain.llms import OpenAI
  
  os.environ["OPENAI_API_KEY"] = 'sk-your-OpenAI-key'
  os.environ['SERPAPI_API_KEY'] = 'your-serpapi-api-key'
  
  LLM = OpenAI(model_name='text-davinci-003',
               temperature=0,
               max_tokens=2048)
  
  tools = load_tools(['serpapi', 'python_repl'])
  # tools = load_tools(['serpapi', 'llm-math'], llm=LLM)
  # tools=load_tools(["serpapi","python_repl"])
  
  agent = initialize_agent(tools=tools,
                           llm=LLM,
                           agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                           verbose=True)
  
  print(agent.run("2023年发售的是什么版本的iPhone?"))
  ```

  代码运行如下

  ```terminal
  > Entering new  chain...
  Retrying langchain.llms.openai.completion_with_retry.<locals>._completion_with_retry in 4.0 seconds as it raised APIConnectionError: Error communicating with OpenAI: ('Connection a
  borted.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None)).
   I need to find out what version of iPhone will be released in 2023
  Action: Search
  Action Input: "iPhone 2023 release date"
  Observation: The iPhone 15 models will likely be released in September 2023 if Apple follows its traditional September launch timeline. Beyond the iPhone 15.
  Thought: I now know the final answer
  Final Answer: The iPhone 15 models will likely be released in September 2023.
  
  > Finished chain.
  The iPhone 15 models will likely be released in September 2023.
  ```

-  简单理解 `initialize_agent` 传入的一些参数

  `tools` 相当于强化学习智能体的动作空间，智能体agent只能用这些设置好的tool完成任务链

  `llm` 相当于强化学习的算法，采用类似于PPO、SAC这些

  `agent` 这个含义不太明确，相当于给这个智能体一个标识

  `verbose` 是布尔变量，为真时在终端输出思考过程；为假时只输出结果。

- 关于agent type 几个选项的含义：

  `zero-shot-react-description`: 根据工具的描述和请求内容的来决定使用哪个工具（最常用）

  `react-docstore`: 使用 ReAct 框架和 docstore 交互, 使用 `Search` 和 `Lookup` 工具, 前者用来搜, 后者寻找term, 举例: `Wipipedia` 工具

  `self-ask-with-search`: 此智能体只使用一个工具: Intermediate Answer, 它会为问题寻找事实答案(指的非 gpt 生成的答案, 而是在网络中,文本中已存在的), 如 `Google search API` 工具

  `conversational-react-description`: 为会话设置而设计的智能体, 它的prompt会被设计的具有会话性, 且还是会使用 ReAct 框架来决定使用来个工具, 并且将过往的会话交互存入内存

#### 项目3. 对文本进行总结

- 文本总结的思考

  对于大预言模型和人一样，对文本进行总结时，文本内容数量要在合适范围内才能做出好的总结。

  当输入长文本时候，大预言模型会报出超过最大token的错误。

  一般思路是：分段/章节/页阅读并总结，然后总结之前生成的总结。这与人的阅读总结过程类似。

- 代码如下

  ```python
  from langchain.chains.summarize import load_summarize_chain
  from langchain.document_loaders import TextLoader
  from langchain.text_splitter import RecursiveCharacterTextSplitter
  from langchain import OpenAI
  import os
  
  os.environ["OPENAI_API_KEY"] = 'sk-your-OpenAI-key'
  
  # 导入文本
  loader = TextLoader("./content.txt")
  # 将文本转成 Document 对象
  document = loader.load()
  print(f'documents:{len(document)}')
  
  # 初始化文本分割器
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                 chunk_overlap=0)
  
  # 切分文本
  splitted_documents = text_splitter.split_documents(document)
  print(f'documents:{len(splitted_documents)}')
  
  # 加载 llm 模型
  LLM_instance = OpenAI(model_name='text-davinci-003',
                        max_tokens=1500)
  
  # 创建总结链
  Chain_instance = load_summarize_chain(llm=LLM_instance,
                                        chain_type='refine',
                                        verbose=False)
  
  # 执行总结链
  response = Chain_instance.run(splitted_documents[:])
  print(response)
  ```

- 结果展示

  ```
  documents:1
  documents:4 
  
  Northeastern University is studying and implementing the speech spirit of General Secretary Xi Jinping and the decisions of the Party Central Committee, emphasizing effectiveness and using Xi Jinping's socialist thought with Chinese characteristics in the new era to mold the foundation and soul, and to inspire and nourish the heart. It is focused on learning ideology and strengthening party spirit, while emphasizing practice and creating new achievements. The school's party committee will deeply study and implement General Secretary Xi Jinping's important speech spirit at the central theme education work conference, fulfill its main responsibility, and organize party members, cadres, teachers, and students to quickly start a learning upsurge with a high sense of political responsibility and mission. Adhering to practical measures, the school will take multiple measures to ensure that the required actions are in place and that optional actions have their own characteristics, making theoretical learning a primary task throughout the theme education and deeply grasping the meaning of "learning to forge the soul" in the theme of education.
  ```

- 代码中的一些参数

  `chunk_overlap` ：切割后的每个 document 里包含几个上一个 document 结尾的内容，主要作用是为了增加每个 document 的上下文关联。

  `chain_type` ：来自作者的文档tutorial

  > 1. `stuff`: 这种最简单粗暴，会把所有的 document 一次全部传给 llm 模型进行总结。如果document很多的话，势必会报超出最大 token 限制的错，所以总结文本的时候一般不会选中这个。
  > 2. `map_reduce`: 这个方式会先将每个 document 进行总结，最后将所有 document 总结出的结果再进行一次总结。
  > 3. `refine`: 这种方式会先总结第一个 document，然后在将第一个 document 总结出的内容和第二个 document 一起发给 llm 模型在进行总结，以此类推。这种方式的好处就是在总结后一个 document 的时候，会带着前一个的 document 进行总结，给需要总结的 document 添加了上下文，增加了总结内容的连贯性。
  > 4. `map_rerank`: 这种一般不会用在总结的 chain 上，而是会用在问答的 chain 上，是一种搜索答案的匹配方式。首先你要给出一个问题，他会根据问题给每个 document 计算一个这个 document 能回答这个问题的概率分数，然后找到分数最高的那个 document ，在通过把这个 document 转化为问题的 prompt 的一部分（问题+document）发送给 llm 模型，最后 llm 模型返回具体答案。

#### 项目4. （Win难）基于本地知识库的问答机器人

代码实现（报了一些错误，建议在Ubuntu下实现）

```python
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA

import os

os.environ["OPENAI_API_KEY"] = 'sk-your-openai-api-key'

# 加载文件夹中的所有txt类型的文件
loader = DirectoryLoader(path="C://Users//aw//PycharmProjects//GPTnew//Contents//", glob='**/*.txt')

# 将数据转成 document 对象，每个文件会作为一个 document
documents = loader.load()

# 初始化分割器
text_splitter = CharacterTextSplitter(chunk_size=200,
                                      chunk_overlap=0)

# 切割加载的 document
split_docs = text_splitter.split_documents(documents)

# 初始化 openai 的 embeddings 对象
embeddings = OpenAIEmbeddings()

# 将 document 通过 openai 的 embeddings 对象计算 embedding 向量信息并临时存入 Chroma 向量数据库，用于后续匹配查询
docsearch = Chroma.from_documents(split_docs, embeddings)

# 创建问答对象
qa = VectorDBQA.from_chain_type(llm=OpenAI(),
                                chain_type="stuff",
                                vectorstore=docsearch,
                                return_source_documents=True)

# 进行问答
question = "本文中付洪源做了什么？"
result = qa({"query": question})
print(result)
```

#### 项目5. （Win难）构建本地持久知识库

- 为什么要构建本地持久知识库

  项目4的embedding在导入数据后需要计算一次；重启这个python就要反复计算，效率就降低了。

  需要将本地文件转换成数据库，最好可以永久放在本地电脑上，这样重复加载时候就会快很多。

- 常见的两种向量数据库，from tutorial

  chroma 是个本地的向量数据库，他提供的一个 `persist_directory` 来设置持久化目录进行持久化。读取时，只需要调取 `from_document` 方法加载即可。

  Pinecone 是一个在线的向量数据库。所以，我可以第一步依旧是注册，然后拿到对应的 api key。https://app.pinecone.io/ （不建议使用这个，需要付费）

- 代码实现（报了一些错误，建议在Ubuntu下实现）

  ```python
  from langchain.embeddings.openai import OpenAIEmbeddings
  from langchain.text_splitter import CharacterTextSplitter
  from langchain import OpenAI, VectorDBQA
  from langchain.document_loaders import DirectoryLoader
  from langchain.vectorstores import Chroma
  
  import os
  
  os.environ["OPENAI_API_KEY"] = 'sk-your-openai-api-key'
  
  # 加载文件夹中的所有txt类型的文件
  loader = DirectoryLoader(path="C://Users//aw//PycharmProjects//GPTnew//Contents//", glob='**/*.txt')
  
  # 将数据转成 document 对象，每个文件会作为一个 document
  documents = loader.load()
  
  # 初始化分割器
  text_splitter = CharacterTextSplitter(chunk_size=200,
                                        chunk_overlap=0)
  
  # 切割加载的 document
  split_docs = text_splitter.split_documents(documents)
  
  # 初始化 openai 的 embeddings 对象
  embeddings = OpenAIEmbeddings()
  
  # 将 document 通过 openai 的 embeddings 对象计算 embedding 向量信息并临时存入 Chroma 向量数据库，用于后续匹配查询
  docsearch = Chroma.from_documents(split_docs, embeddings,
                                    persist_directory="C://Users//aw//PycharmProjects//GPTnew//Contents//")
  docsearch.persist()
  
  # 加载数据
  docsearch = Chroma(persist_directory="C://Users//aw//PycharmProjects//GPTnew//Contents//",
                     embedding_function=embeddings)
  
  # 创建问答对象
  qa = VectorDBQA.from_chain_type(llm=OpenAI(),
                                  chain_type="stuff",
                                  vectorstore=docsearch,
                                  return_source_documents=True)
  
  # 进行问答
  question = "本文中付洪源做了什么？"
  result = qa({"query": question})
  print(result)
  ```

#### 项目6. （Win难）使用GPT-3.5模型构建基于youtube网站的视频问答机器人

代码如下（报了一些错误，建议在Ubuntu下实现）

```python
import os

from langchain.document_loaders import YoutubeLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ChatVectorDBChain, ConversationalRetrievalChain

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

os.environ["OPENAI_API_KEY"] = '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'

# 加载 youtube 频道
loader = YoutubeLoader.from_youtube_url(
    youtube_url='https://www.youtube.com/watch?v=DsYDmg72K1k&list=PLfVqr2l0FG-u7chWKPQMDoT0o-I2ejxeK&index=1')

# 将数据转成 document
document = loader.load()

# 初始化文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=20
)

# 分割 youtube documents
documents = text_splitter.split_documents(document)

# 初始化 openai embeddings
embeddings = OpenAIEmbeddings()

# 将数据存入向量存储
vector_datastore = Chroma.from_documents(documents, embeddings)
# 通过向量存储初始化检索器
retriever = vector_datastore.as_retriever()

system_template = """
Use the following context to answer the user's question.
If you don't know the answer, say you don't, don't try to make it up. And answer in Chinese.
-----------
{context}
-----------
{chat_history}
"""

# 构建初始 messages 列表，这里可以理解为是 openai 传入的 messages 参数
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template('{question}')
]

# 初始化 prompt 对象
prompt = ChatPromptTemplate.from_messages(messages)

# 初始化问答链
qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.1, max_tokens=2048),
                                           retriever,
                                           condense_question_prompt=prompt)

chat_history = []
while True:
    question = input('问题：')
    # 开始发送问题 chat_history 为必须参数,用于存储对话历史
    result = qa({'question': question, 'chat_history': chat_history})
    chat_history.append((question, result['answer']))
    print(result['answer'])
```

---

#### 项目7. 执行多个大语言链

```python
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain

import os

os.environ["OPENAI_API_KEY"] = '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'

llm = OpenAI(temperature=1)

# location 链
template = """
Your job is to come up with a classic dish from the area that the users suggests.
% USER LOCATION
{user_location}

YOUR RESPONSE:
"""

prompt_template = PromptTemplate(input_variables=['user_location'], template=template)

location_chain = LLMChain(llm=llm, prompt=prompt_template)

# meal 链
template = """Given a meal, give a short and simple recipe on how to make that dish at home.
% MEAL
{user_meal}

YOUR RESPONSE:
"""

prompt_template = PromptTemplate(input_variables=["user_meal"], template=template)

meal_chain = LLMChain(llm=llm, prompt=prompt_template)
# 通过 SimpleSequentialChain 串联起来，第一个答案会被替换第二个中的user_meal，然后再进行询问
overall_chain = SimpleSequentialChain(chains=[location_chain, meal_chain], verbose=True)
review = overall_chain.run("Rome")
```

执行结果：

```
> Entering new  chain...
One classic dish from Rome is spaghetti alla carbonara. It is a traditional Italian pasta dish made with eggs, Parmigiano-Reggiano cheese, guanciale, and black pepper.
Ingredients:
- 8oz spaghetti
- 2 eggs, lightly beaten
- 2 tablespoons butter
- 1/4 cup grated Parmigiano-Reggiano cheese
- 4 ounces guanciale, diced
- Black pepper to taste

Instructions:
1. Cook the spaghetti in boiling salted water until al dente and drain.
2. In a bowl, mix together the eggs, butter, Parmigiano-Reggiano cheese, and diced guanciale.
3. Place the spaghetti in a skillet on medium heat and add the egg mixture.
4. Mix together and cook until the eggs thicken and lightly scramble.
5. Add black pepper to taste.
6. Serve the spaghetti alla carbonara hot. Enjoy!

> Finished chain.
```

#### 项目8. 结构化输出

```python
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

import os

os.environ["OPENAI_API_KEY"] = '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'

llm = OpenAI(model_name="text-davinci-003")

# 告诉他我们生成的内容需要哪些字段，每个字段类型式啥
response_schemas = [
    ResponseSchema(name="bad_string", description="This a poorly formatted user input string"),
    ResponseSchema(name="good_string", description="This is your response, a reformatted response")
]

# 初始化解析器
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# 生成的格式提示符
# {
#	"bad_string": string  // This a poorly formatted user input string
#	"good_string": string  // This is your response, a reformatted response
#}
format_instructions = output_parser.get_format_instructions()

template = """
You will be given a poorly formatted string from a user.
Reformat it and make sure all the words are spelled correctly

{format_instructions}

% USER INPUT:
{user_input}

YOUR RESPONSE:
"""

# 将我们的格式描述嵌入到 prompt 中去，告诉 llm 我们需要他输出什么样格式的内容
prompt = PromptTemplate(
    input_variables=["user_input"],
    partial_variables={"format_instructions": format_instructions},
    template=template
)

promptValue = prompt.format(user_input="welcom to califonya!")
llm_output = llm(promptValue)

# 使用解析器进行解析生成的内容
response = output_parser.parse(llm_output)
print(response)
```

结果展示

```
{'bad_string': 'welcom to califonya!', 'good_string': 'Welcome to California!'}
```

#### 项目9. 爬取网页并输出JSON数据

代码展示

```python
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMRequestsChain, LLMChain

import os

os.environ["OPENAI_API_KEY"] = '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'

llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0)

template = """在 >>> 和 <<< 之间是网页的返回的HTML内容。
网页是新浪财经A股上市公司的公司简介。
请抽取参数请求的信息。

>>> {requests_result} <<<
请使用如下的JSON格式返回数据
{{
  "company_name":"a",
  "company_english_name":"b",
  "issue_price":"c",
  "date_of_establishment":"d",
  "registered_capital":"e",
  "office_address":"f",
  "Company_profile":"g"

}}
Extracted:"""

prompt = PromptTemplate(
    input_variables=["requests_result"],
    template=template
)

chain = LLMRequestsChain(llm_chain=LLMChain(llm=llm, prompt=prompt))
inputs = {
  "url": "https://vip.stock.finance.sina.com.cn/corp/go.php/vCI_CorpInfo/stockid/600519.phtml"
}

response = chain(inputs)
print(response['output'])
```

结果展示

```
C:\Users\aw\.conda\envs\GPTnew\Lib\site-packages\langchain\llms\openai.py:179: UserWarning: You are trying to use a chat model. This way of initializing it is no longer supported. 
Instead, please use: `from langchain.chat_models import ChatOpenAI`
  warnings.warn(
C:\Users\aw\.conda\envs\GPTnew\Lib\site-packages\langchain\llms\openai.py:751: UserWarning: You are trying to use a chat model. This way of initializing it is no longer supported. 
Instead, please use: `from langchain.chat_models import ChatOpenAI`
  warnings.warn(
{
  "company_name":"贵州茅台酒股份有限公司",
  "company_english_name":"Kweichow Moutai Co.,Ltd.",
  "issue_price":"31.39",
  "date_of_establishment":"1999-11-20",
  "registered_capital":"125620万元(CNY)",
  "office_address":"贵州省仁怀市茅台镇",
  "Company_profile":"公司是根据贵州省人民政府黔府函〔1999〕291号文,由中国贵州茅台酒厂有限责任公司作为主发起人,联合贵州茅台酒厂技术开发公司、贵州省轻纺集体工业联社、深圳清华大学研究
院、中国食品发酵工业研究院、北京市糖业烟酒公司、江苏省糖烟酒总公司、上海捷强烟草糖酒(集团)有限公司于1999年11月20日共同发起设立的股份有限公司。经中国证监会证监发行字[2001]41号文核准
并按照财政部企[2001]56号文件的批复,公司于2001年7月31日在上海证券交易所公开发行7,150万(其中,国有股存量发行650万股)A股股票。主营业务：贵州茅台酒系列产品的生产与销售,饮料、食品、包装 
材料的生产与销售,防伪技术开发;信息产业相关产品的研制和开发等。"
}
```

####  项目9: 自定义agent中所使用的工具

自定义工具里面有个比较有意思的地方，使用哪个工具的权重是**靠工具中描述内容**来实现的，和我们之前编程靠数值来控制权重完全不同。

```python
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.tools import BaseTool
from langchain.llms import OpenAI
from langchain import LLMMathChain, SerpAPIWrapper

llm = OpenAI(temperature=0)

# 初始化搜索链和计算链
search = SerpAPIWrapper()
llm_math_chain = LLMMathChain(llm=llm, verbose=True)

# 创建一个功能列表，指明这个 agent 里面都有哪些可用工具
tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math"
    )
]

# 初始化 agent
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# 执行 agent
agent.run("Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?")
```

结果展示

```
> Entering new  chain...
 I need to find out who Leo DiCaprio's girlfriend is and then calculate her age raised to the 0.43 power.
Action: Search
Action Input: "Leo DiCaprio girlfriend"
Observation: Leonardo DiCaprio and Gigi Hadid were recently spotted at a pre-Oscars party, sparking interest once again in their rumored romance. The Revenant actor and the model f
irst made headlines when they were spotted together at a New York Fashion Week afterparty in September 2022.
Thought: I need to find out Gigi Hadid's age.
Action: Search
Action Input: "Gigi Hadid age"
Observation: 28 years
Thought: I now know the age and can calculate the result.
Action: Calculator
Action Input: 28^0.43

> Entering new  chain...
28^0.43```text
28**0.43
​```
...numexpr.evaluate("28**0.43")...

Answer: 4.1906168361987195
> Finished chain.

Observation: Answer: 4.1906168361987195
Thought: I now know the final answer.
Final Answer: Gigi Hadid's age raised to the 0.43 power is 4.1906168361987195.

> Finished chain.
```

#### 项目10. 使用Memory实现一个带记忆的对话机器人

代码实现

```python
from langchain.memory import ChatMessageHistory
from langchain.chat_models import ChatOpenAI

import os

os.environ["OPENAI_API_KEY"] = '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'

chat = ChatOpenAI(temperature=1)

history = ChatMessageHistory()

# 给 MessageHistory 对象添加对话内容
history.add_ai_message("你好！")
history.add_user_message("中国的首都是哪里？")

# 执行对话
ai_response = chat(history.messages)
print(ai_response.content)
```

执行结果

```
中国的首都是北京。
```

### 实践时候报错实录

#### 1. 输入OpenAI的密钥名字不对。

```
Did not find openai_api_key, please add an environment variable `OPENAI_API_KEY` which contains it, or pass  `openai_api_key` as a named parameter. (type=value_error)
```

Solve：不能输入成 `OPENAI-API-KEY` ，应该要输入成 `OPENAI_API_KEY` ，才能被识别。

#### 2. 运行代码的时候正好遇上了高峰期。

```
requests.exceptions.ConnectionError: ('Connection aborted.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None))
```

Solve：简单粗暴方法：直接再运行一次。

#### 3. 没有 `unstructured` 这个python库

```
ValueError: unstructured package not found, please install it with `pip install unstructured`
```

Solve：`pip install unstructured`

#### 4. `UnstructuredFileLoader` 的实例无法执行 `.load()` 方法

```
Process finished with exit code -1073741819 (0xC0000005)
```

- Solve：根据文本类型手动选择载入器，比如文本是 `.txt` 文件，就选用 `TextLoader`

---

```
UnicodeDecodeError: 'gbk' codec can't decode byte 0xa2 in position 52: illegal multibyte sequence
```

- Solve：文本中避免出现中文字符。

#### 5. `DirectoryLoader` 载入文件路径报错

```
SyntaxError: (unicode error) 'unicodeescape' codec can't decode bytes in position 2-3: truncated \UXXXXXXXX escape
```

Solve：Win系统下的文件路径输入和Linux不一样，要把 `\` 路径分割符换成 `//` ；最好使用全局路径

#### 6. 没有 `chromadb` 这个python库

```
ValueError: Could not import chromadb python package. Please install it with `pip install chromadb`.
```

Solve：在Windows下使用很麻烦，建议转到Ubuntu下使用

#### 7. 没有 `youtube-transcript-api` 这个python库

```
ImportError: Could not import youtube_transcript_api python package. Please install it with `pip install youtube-transcript-api`.
```

Solve：`pip install youtube-transcript-api`











