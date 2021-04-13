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
模型训练的特征。请选择标称型数据字符型

label
标签
模型训练的标签。选择分类标签所在的列，仅支持一般为字符型数据。

add_col
新增预测列列名
predict_label   [默认值]
"""

# 基础参数
"""
criterion
切分时的评价准则
entropy
切分时的评价准则

splitter
切分原则
best
切分原则
"""

# 高级参数
"""
max_depth
最大深度
None
树的最大深度。如果为None，则扩展节点直到     所有叶子都是纯净的，或者直到所有叶子都含有少于     min_samples_split样本。

min_samples_split
拆分内部节点所需的最小样本数
2
拆分内部节点所需的最小样本数,大于0的整数类型

min_samples_leaf
叶子节点所需的最小样本数
1
叶子节点所需的最小样本数，大于0的整数类型

min_weight_fraction_leaf
权重总和的最小加权分数
0
权重总和的最小加权分数，非负数（0，正整数）
"""



def execute(conn, inputs, params, outputs, reportFileName):
    #<editable>
	'''
	载入模块
	'''
	import pyh
	import report_utils
	import db_utils
	import numpy as np
	import pandas as pd
	from sklearn.tree import DecisionTreeClassifier as dtc
	from sklearn.metrics import confusion_matrix # 导入混淆矩阵函数
	from sklearn.metrics import  precision_recall_curve, roc_curve, auc
	from sklearn.preprocessing import label_binarize
	from sklearn.metrics import classification_report
	import matplotlib.pyplot as plt
	from itertools import cycle
	from sklearn import metrics
	from sklearn.externals.six import StringIO
	from sklearn import tree
	import pydot
	import warnings
	warnings.filterwarnings("ignore")
	report = report_utils.Report()

	'''
	选择目标数据
	'''
	data_in = db_utils.query(conn, 'select '+ params['features'] + ',' + params['label'] + ' from ' + inputs['data_in'])
	x_train = data_in.drop(params['label'], 1)
	y_train = data_in[params['label']]
	y_train=y_train.astype(str)
	class_names = y_train.unique()
	n_classes = len(class_names)
	y_one_hot = label_binarize(y_train, classes =class_names)#装换成类似二进制的编码
	y_binarize = label_binarize(y_train, classes = class_names) # 标签热编码

	'''
	构建ID3树模型

	'''
	if(type(params['max_depth'])!=int):
	    #params['max_depth']='None'
	    model =dtc(criterion=params['criterion'],splitter=params['splitter'],max_depth=None,min_samples_split=int(params['min_samples_split']),min_samples_leaf=int(params['min_samples_leaf']),min_weight_fraction_leaf=int(params['min_weight_fraction_leaf']))
	else:
	    model =dtc(criterion=params['criterion'],splitter=params['splitter'],max_depth=int(params['max_depth']),min_samples_split=int(params['min_samples_split']),min_samples_leaf=int(params['min_samples_leaf']),min_weight_fraction_leaf=int(params['min_weight_fraction_leaf']))

	'''
	模型训练
	'''
	structure=model.fit(x_train, y_train) # 训练

	'''
	报告
	'''
	report.h1('id3算法')
	model_params={}
	model_params['模型分类']=model.n_classes_
	model_params['模型特征数']=model.n_features_
	a=pd.DataFrame([model_params.keys(),model_params.values()]).T
	a.columns=['参数名称','参数值']
	report.h3('模型属性')
	report.p("输出模型的属性信息。")
	report.table(a)

	x_train.columns
	a=pd.DataFrame([x_train.columns,np.around(model.feature_importances_,decimals=4)]).T
	a.columns=['特征','feature importance']
	a=a.sort_values('feature importance',ascending=False)
	report.p("        ")
	report.p("特征重要性如下：")
	report.table(a)
	a.index=[a['特征'].unique()]

	#显示所有列
	pd.set_option('display.max_columns', None)
	#显示所有行
	pd.set_option('display.max_rows', None)
	#设置value的显示长度为100，默认为50
	pd.set_option('max_colwidth',100)

	a.plot(kind='barh',figsize=(10,6),).get_figure()
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.ylabel('')
	plt.tight_layout()
	plt.legend(loc='best')
	plt.savefig('bar')
	plt.show()
	report.image('bar.png')

	cm = confusion_matrix(y_train, fit_label) # 混淆矩阵
	n_classes = len(cm)
	n_classes
	if(n_classes == 2):
	    cm = confusion_matrix(y_train, fit_label) # 混淆矩阵
	    TP=cm[0][0]
	    FN=cm[0][1]
	    FP=cm[1][0]
	    TN=cm[1][1]
	    acc=(TP+TN)/(TP+TN+FP+FN)
	    precision=TP/(TP+FP)
	    recall=TP/(TP+FN)
	    f1=2*precision*recall/(precision+recall)
	    model_params={}
	    model_params['accuracy']=np.around(acc,decimals=2)
	    model_params['precision']=np.around(precision,decimals=2)
	    model_params['recall']=np.around(recall,decimals=2)
	    model_params['f1']=np.around(f1,decimals=2)

	    a=pd.DataFrame([model_params.keys(),model_params.values()]).T
	    a.columns=['指标','值']

	    report.h3('模型评价指标')
	    report.table(a)
	    print(acc)
	    print(precision)
	    print(recall)
	    print(f1)

	if(n_classes >2):
		from sklearn import preprocessing
		import numpy as np
		binarizer = preprocessing.Binarizer(threshold=0.5)
		y_score=binarizer.transform(y_score)
		target_names = class_names
		a=classification_report(y_train, fit_label,target_names=target_names)
		b=a.split('\n')
		res=[]
		for bb in b:
			if(bb!=''):
				z=[]
				c=bb.split('  ')
				for cc in c:
					if(cc!=''):
						z.append(cc.strip())
				res.append(z)
		res_table=pd.DataFrame(res[1:])
		res_table.columns=['index','precision', 'recall', 'f1-score', 'support']
		report.h3('模型评价指标')
		report.table(res_table)

	'''
	绘制混淆矩阵图
	'''
	cm = confusion_matrix(y_train, fit_label) # 混淆矩阵
	plt.figure(figsize=(4, 4))
	plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
	for x in range(len(cm)):
		for y in range(len(cm)):
			plt.annotate(cm[x,y], xy=(y, x),
						 size = 'large',
						 horizontalalignment='center',
						 verticalalignment='center')

	tick_marks = np.arange(len(class_names))
	plt.xticks(tick_marks, class_names)
	plt.yticks(tick_marks, class_names)
	plt.xlabel('Predict Label') # 坐标轴标签
	plt.ylabel('True Label') # 坐标轴标签
	plt.tight_layout()
	plt.savefig('cm_img.png')
	plt.show()
	report.h3('混淆矩阵')
	report.p("如下所示混淆矩阵图：")
	report.image('cm_img.png')

	dot_data = StringIO()
	tree.export_graphviz(model, out_file=dot_data)
	graph = pydot.graph_from_dot_data(dot_data.getvalue())
	graph[0].write_dot('model.dot')
	graph[0].write_png('model.png')
	report.h3('id3图')
	report.image('model.png')

	'''
	绘制ROC曲线
	fpr：假正例率
	tpr：真正例率
	'''
	colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
	fit_label=label_binarize(fit_label, classes = class_names)
	if(n_classes == 2):
		fpr, tpr, _ = roc_curve(y_binarize.ravel(),fit_label.ravel())
		roc_auc = auc(fpr, tpr)
		plt.figure(figsize=(8, 4))
		lw = 2
		plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
		plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
		plt.fill_between(fpr, tpr,alpha=0.2,color='b')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('ROC and AUC')
		plt.legend(loc="lower right")
		plt.savefig('roc.png')
		plt.show()
		report.h3('ROC图')
		report.p("如下图所示：AUC所占的面积是"+str(np.around(roc_auc,decimals=2)))
		report.image('roc.png')

	if(n_classes == 2):
		fpr, tpr, _ = precision_recall_curve(y_binarize.ravel(),fit_label.ravel())
		roc_auc = auc(fpr, tpr)
		fpr[0]=0
		plt.figure(figsize=(8, 4))
		lw = 2
		plt.plot(fpr, tpr, label='PR')
		plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
		plt.fill_between(fpr, tpr,alpha=0.2,color='b')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.0])
		plt.xlabel('precision')
		plt.ylabel('recall')
		plt.title('PR')
		plt.legend(loc="lower right")
		plt.savefig('pr.png')
		plt.show()
		report.h3('Precision-Recall图')
		report.image('pr.png')

	if(n_classes >2):
		#print('调用函数auc：', metrics.roc_auc_score(y_binarize, y_score, average='micro'))
		# 2、手动计算micro类型的AUC
		#首先将矩阵y_one_hot和y_score展开，然后计算假正例率FPR和真正例率TPR
		fpr, tpr, thresholds = metrics.roc_curve(y_binarize.ravel(),fit_label.ravel())
		auc = metrics.auc(fpr, tpr)
		print('手动计算auc：', auc)    #绘图
		plt.figure(figsize=(8, 4))
		lw = 2
		plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
		plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
		plt.fill_between(fpr, tpr,alpha=0.2,color='b')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('ROC and AUC')
		plt.legend(loc="lower right")
		plt.savefig('roc.png')
		plt.show()
		report.h3('ROC图')
		report.p("如下图所示：AUC所占的面积是"+str(np.around(auc,decimals=2)))
		report.image('roc.png')

	report.writeToHtml(reportFileName)

	'''
	将结果写出
	'''
	db_utils.dbWriteTable(conn, outputs['data_out'], data_out)
	return(model)
    #</editable>
