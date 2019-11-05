from jieba import lcut
import numpy as np
from gensim.similarities import SparseMatrixSimilarity
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
# 文本集和搜索词
texts = [
    '关爱幼儿，尊重每个幼儿的人格尊严与基本权利。理解幼儿教育在人一生发展中的重要性，能认识到幼儿教育必须以每一个幼儿的全面发展为本。 理解教师职业的光荣与责任，具有从事幼儿教育工作的热情。了解幼儿教师专业发展的要求，具有终身学习与自主发展的意识。了解国家主要的教育法律法规，了解《儿童权利公约》。 熟悉教师职业道德规范，能评析保育教育实践中的道德规范问题。了解幼儿园教师的职业特点与职业行为规范，能自觉地约束自己的职业行为。 有爱心、耐心、责任心。了解自然和人文社会科学的一般知识，熟悉常见的幼儿科普读物和文学作品，具有较好的文化修养。 具有较好的艺术修养和审美能力。 具有较好的人际交往与沟通能力。 具有一定的阅读理解能力、语言与文字表达能力、信息获得与处理能力。',
    '了解婴幼儿发展的基本原理。了解婴幼儿生理与心理发展的基本规律，熟悉幼儿身体发育、动作发展和认知、情绪情感、个性、社会性发展的特点。了解幼儿发展中的个体差异及其形成原因，能运用相关知识分析教育中的有关问题。了解研究幼儿的基本方法，并能据此初步了解幼儿的发展状况和教育需求。了解幼儿发展中易出现的问题或障碍。掌握教育的基本理论，并能据此分析教育现象与问题。掌握学前教育的基本理论，并能据此分析学前教育中的现象与问题。了解幼教发展简史和著名教育家的儿童教育思想，并能结合幼教的现实问题进行分析。掌握幼儿教育的基本原则和不同于中小学教育的基本特点，并能据此评析幼教实践中的问题。理解幼儿游戏的意义与作用。理解幼儿园环境创设、班级管理的目的和意义。熟悉《幼儿园教育指导纲要（试行）》，了解幼教改革动态。',
    '熟悉幼儿园一日生活的主要环节，具有将教育融入一日生活的意识。了解幼儿生活常规教育的内容和要求以及培养幼儿良好生活、卫生习惯的方法。了解幼儿保健、安全方面的基本知识和处理常见问题与突发事件的基本方法。熟悉幼儿园环境创设的原则与基本方法。理解教师的态度、言行对幼儿园心理环境形成中的重要性，并能进行自我调控。了解幼儿园常见活动区的功能，能根据幼儿的需要创设相应的活动区。理解协调家庭、社区等各种教育力量的重要性，了解与家长沟通与交流的基本方法。熟悉幼儿游戏的类型及其各类游戏的特点和主要功能。了解各年龄阶段幼儿的游戏特点，能根据需要提供支持与指导。能根据教育目标和幼儿的兴趣需要和年龄特点选择教育内容，确定活动目标，设计教育活动方案。掌握幼儿健康、语言、社会、科学、艺术等领域教育的基本知识和相应的教育方法理解各领域之间的联系和开展综合教育活动的意义与方法。活动过程中关注幼儿的表现和反应，并能据此进行调整。关注个体差异，能根据幼儿的个体需要给予指导。了解幼儿园教育评价的目的与方法，能对保教工作进行评价与反思。能正确运用评价结果改进保教工作，促进幼儿发展']
# 标准
keyword = '对给定的主题或情况提出不同寻常的或聪明的想法的能力，或用创造性的方法来解决问题。进行作曲、创作、音乐、舞蹈、视觉艺术、戏剧和雕塑作品所需的理论知识和技巧。了解他人的反应，理解他们为什么会有这样的反应。把他人聚在一起，'
# 1、将【文本集】生成【分词列表】
texts = [lcut(text) for text in texts]
with open ('tidaixing.txt','r') as f:
    lines = f.readlines()
text_julei = ''.join(lines)
text_fenci = lcut(text_julei)
text_chuliwan = ' '.join(text_fenci)
with open ('tidaixing_chuliwan.txt','w') as f:
    f.writelines(text_chuliwan)
'''


# 2、基于文本集建立【词典】，并获得词典特征数
dictionary = Dictionary(texts)
for each in dictionary:
    print(each,dictionary[each])
num_features = len(dictionary.token2id)
# 3.1、基于词典，将【分词列表集】转换成【稀疏向量集】，称作【语料库】
corpus = [dictionary.doc2bow(text) for text in texts]
# 3.2、同理，用【词典】把【搜索词】也转换为【稀疏向量】
kw_vector = dictionary.doc2bow(lcut(keyword))
# 4、创建【TF-IDF模型】，传入【语料库】来训练
tfidf = TfidfModel(corpus)
# 5、用训练好的【TF-IDF模型】处理【被检索文本】和【搜索词】
tf_texts = tfidf[corpus]  # 此处将【语料库】用作【被检索文本】
tf_kw = tfidf[kw_vector]
# 6、相似度计算
sparse_matrix = SparseMatrixSimilarity(tf_texts, num_features)
similarities = sparse_matrix.get_similarities(tf_kw)
score = []
for e, s in enumerate(similarities, 1):
    print('kw 与 text%d 相似度为：%.2f' % (e, s))
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
print(np.exp(similarities)/sum(np.exp(similarities)))
'''