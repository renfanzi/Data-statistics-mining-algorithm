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
features
特征
模型训练的特征。请选择数值型数据

label
标签
模型训练的标签。请选择数值型数据

add_col
新列名
predict_value   [默认值]
新增的预测列的列名
"""

# 基础参数
"""
n_neighbors
最近邻点数
5
每个点的邻居的数目

weights
权重类型
uniform
用于预测的权重函数

p
闵可夫斯基距离参数
2
闵可夫斯基距离参数
"""

# 高级参数
"""
algorithm
算法
auto
用于计算最近邻的算法。

leaf_size
叶子数
30
传递给球树或KD树的叶子大小。这可能会影响构造和查询的速度，以及存储树所需的内存。最佳值取决于问题的性质。

metric
距离度量
minkowski
用于距离计算的度量。
"""



def execute(conn, inputs, params, outputs, reportFileName):
    #<editable>
	'''
	导入模块
	'''
	import numpy as np
	import matplotlib.pyplot as plt
	import pandas as pd
	import itertools
	from sklearn.neighbors import KNeighborsRegressor
	from sklearn.metrics import explained_variance_score,mean_squared_error,mean_absolute_error,r2_score
	import pyh
	import report_utils
	import db_utils

	'''
	载入数据
	'''
	X_train = db_utils.query(conn, 'select '+ params['features'] + ' from ' + inputs['data_in'])
	y_train = db_utils.query(conn, 'select '+ params['label'] + ' from ' + inputs['data_in'])

	'''
	建立模型
	'''
	knn = KNeighborsRegressor( n_neighbors=int(params['n_neighbors']), weights=params['weights'], algorithm=params['algorithm'], leaf_size=int(params['leaf_size']), metric=params['metric'], p=int(params['p']),n_jobs=1,metric_params=None)

	'''
	模型训练与拟合
	'''
	model = knn.fit(X_train, y_train)
	y_ov=model.predict(X_train)

	'''
	模型参数
	'''
	pars=pd.DataFrame([['最近邻点数','权重类型','算法','叶子数','距离度量','闵可夫斯基距离参数'],[int(params['n_neighbors']),params['weights'],params['algorithm'],int(params['leaf_size']),params['metric'],int(params['p'])]]).T
	pars.columns=['参数名称','参数值']
	report.h3('模型参数')
	report.p("需要配置的参数及其取值如下。")
	report.table(pars)

	'''
	模型指标
	'''
	report.h3('模型评价指标')
	report.p('模型拟合效果指标如下。')
	md_info=pd.DataFrame()
	md_info['重要指标']=['MSE','RMSE','MAE','EVS','R-Squared']
	md_info['值']=['%.2f'%mean_squared_error(y_train, y_ov),'%.2f'%np.sqrt(mean_squared_error(y_train, y_ov)),'%.2f'%mean_absolute_error(y_train,y_ov),'%.2f'%explained_variance_score(y_train, y_ov),'%.2f'%r2_score(y_train, y_ov)]
	report.table(md_info)

	'''
	模型拟合情况
	'''
	data_out=pd.concat([X_train,y_train,pd.DataFrame({params['add_col']:list(itertools.chain.from_iterable(y_ov))})],axis=1)
	data_out[params['add_col']]=data_out.apply(lambda x:'%.2f'%x[params['add_col']],axis=1)
	report.h3('模型拟合情况')
	report.p('取'+str(len(X_train))+'条数据作为训练集，建立最近邻回归模型，得到的预测值与真实值对比图如下图所示。')
	X_label=[str(i) for i in X_train.index]
	plt.style.use('ggplot')
	plt.figure(figsize=(6.0, 4.0))
	plt.plot(X_label, y_train, marker='*',label='origin')
	plt.plot(X_label, y_ov, marker='.',alpha=0.7,label='prediction')
	if len(X_train)>10 and (len(X_train)-1)%10<5:plt.xticks(np.linspace(0,np.ceil(len(X_train)/5)*5-1,5))
	elif len(X_train)>10 and (len(X_train)-1)%10>5:plt.xticks(np.linspace(0,np.ceil(len(X_train)/10)*10-1,10))
	else:plt.xticks(np.linspace(0,len(X_train)-1,len(X_train)))
	plt.legend()
	plt.title('Fitting of Nearest Neighbors Regression models')
	plt.xlabel('index')
	plt.ylabel(params['label'])
	plt.tight_layout()
	plt.savefig('overview.png')
	report.image('overview.png')

	#保存模型
	modelFile = 'model.pkl'
	joblib.dump(model, modelFile)

	'''
	生成报告
	'''
	report.writeToHtml(reportFileName)

	'''
	将结果写出
	'''
	db_utils.dbWriteTable(conn, outputs['data_out'], data_out)
	return(model)
    #</editable>
