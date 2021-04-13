# <editable>
# 在此处添加算法描述
# </editable>
# conn：             数据库连接
# inputs：           输入数据集合，数据类型：list， 存储组件输入节点对应的数据，
#                    通过输入节点的key获取数据，例如配置的key为“input1”, 那么inputs$input1
#                    即为该节点对应的数据表
# params：           参数集合，数据类型：list， 存储，获取的规则与inputs一致。需要注意的是：
#                    params中参数的值都是字符类型的，需要在代码中进行数据类型转换，比如：
#                    as.integer(params$centers)
# outputs：          存储规则参见inputs
# reportFileName：   算法运行报告文件的存储路径
# 返回值(可选)：     如果函数用于训练模型，则必须返回模型对象

# 字段属性
"""
columns
特征
模型训练的特征。请选择数值型数据，如果勾选了非数值类型数据，则会自动过滤，下个组件可能无法获取所有列。

add_col
定义聚类列列名
cluster_id  [默认值]
"""

# 基础参数
"""
n_clusters
聚类数	
2
模型聚类结果的类别数目。大于0的整数
"""

# 高级参数
"""
affinity
affinity
euclidean
用于计算链接的度量标准。可以是“欧几里德”，“l1”，“l2”，     “曼哈顿”，“余弦”。 如果linkage是“ward”，则只接受“欧几里德”

linkage
linkage
average     [默认值]
"""




def execute(conn, inputs, params, outputs, reportFileName):
    #<editable>
	'''
	载入模块
	'''
	from sklearn.cluster import AgglomerativeClustering
	import numpy as np
	import pandas as pd
	import matplotlib.pyplot as plt
	from collections import Counter
	from sklearn.manifold import TSNE
	import pyh
	import report_utils
	import db_utils
	import warnings
	warnings.filterwarnings('ignore')
	report = report_utils.Report()

	'''
	选择目标数据
	'''
	data_in = db_utils.query(conn, 'select '+ params['columns'] + ' from ' + inputs['data_in'])

	'''
	报告
	'''
	report.h1('层次聚类')
	model_params={}
	model_params['K值']=params['n_clusters']
	model_params['距离度量标准']=params['affinity']
	model_params['linkage']=params['linkage']
	a=pd.DataFrame([model_params.keys(),model_params.values()]).T
	a.columns=['参数名称','参数值']
	report.h3('模型参数')
	report.p("输出配置的参数以及参数的取值。")
	report.table(a)

	'''
	层次聚类
	'''
	data_in = data_in.select_dtypes(include=['number'])  # 筛选数值型数据
	clst = AgglomerativeClustering(n_clusters = int(params['n_clusters']),affinity= params['affinity'],linkage= params['linkage'])

	'''
	预测聚类标签并输出
	'''
	fit_label = clst.fit_predict(data_in) + 1
	data_out = np.column_stack((data_in, fit_label))
	# 转换成pd.DataFrame
	columns = data_in.columns
	columns = np.append(columns, params['add_col'])
	data_out = pd.DataFrame(data_out, columns = columns)
	model_params={}
	model_params['n_components_']=clst.n_components_
	model_params['n_leaves_']=clst.n_leaves_
	a=pd.DataFrame([model_params.keys(),model_params.values()]).T
	a.columns=['参数名称','参数值']
	report.h3('模型属性')
	report.p("输出模型的属性信息。")
	report.table(a)
	lable_res=pd.Series(fit_label)
	lable_res=lable_res.value_counts()
	lable_res=pd.DataFrame(lable_res)
	lable_res['group']=lable_res.index
	lable_res['group']=lable_res['group'].apply(lambda x:'group'+str(x))
	lable_res=lable_res.reset_index(drop=True)
	fig = plt.figure(figsize=(8,6))
	a=pd.DataFrame([lable_res['group'],lable_res[0]])
	b=[]
	for i in range(len(lable_res)):
	    b.append(0.1)
	labels = pd.Series(lable_res['group']).unique()
	fracs = list(Counter(lable_res[0]))
	plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
	plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
	plt.axes(aspect=1)
	plt.pie(x=fracs,labels=labels,
	        explode=tuple(b),
	        autopct='%3.1f %%',
	        shadow=True, labeldistance=1.1,
	        startangle = 180,
	        pctdistance = 0.6,
	        radius=2.5)
	plt.axis('equal')
	plt.title('pie')
	plt.legend(loc=0, bbox_to_anchor=(0.92, 1))  # 图例
	#设置legend的字体大小
	leg = plt.gca().get_legend()
	ltext = leg.get_texts()
	plt.setp(ltext, fontsize=6)
	plt.savefig("pie.png")
	plt.show()

	'''
	报告
	'''
	report.h3('饼图结果概况')
	string='由饼图可以看出，总共分为'+str(len(lable_res))+'个聚类分群，分别是：'
	for i in range(len(lable_res)):
	    if(i<len(lable_res)-1):
	        string=string+lable_res.loc[i][1]+"、"
	    else:
	        string=string+lable_res.loc[i][1]+"。"
	report.p(string)
	for i in range(len(lable_res)):
	    report.p(lable_res.loc[i][1]+"的个数为"+str(lable_res.loc[i][0])+"，占比为"+str(np.around(lable_res.loc[i][0]/lable_res[0].sum(),decimals=2)))
	report.image('pie.png')
	out=data_in
	out['label']=fit_label
	tsne = TSNE(n_components=2, learning_rate=100).fit_transform(out)
	fig = plt.figure()
	x=tsne[:,0]
	y=tsne[:,1]
	plt.scatter(x, y, c=out['label'])
	plt.title("scatter\n");
	plt.savefig('scatter.png')
	plt.show()
	report.h3('散点图示例')
	report.p('通过对数据进行降维，在二维空间中展示的聚类结果。')
	report.image('scatter.png')
	report.writeToHtml(reportFileName)

	'''
	将结果写出
	'''
	db_utils.dbWriteTable(conn, outputs['data_out'], data_out)
    #</editable>
