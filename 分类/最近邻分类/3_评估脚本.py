# <editable>
# 在此处添加算法描述
# </editable>
# conn：             数据库连接
# model：            评估的模型
# inputs:            输入集合
# params:            参数
# outputs:           输出集合
# reportFileName：   评估报告文件的存储路径
def evaluate(conn, model, inputs, params, outputs, reportFileName):
    #<editable>
	'''
	载入模块
	'''
	import pyh
	import report_utils
	import db_utils
	import numpy as np
	import pandas as pd
	from sklearn.metrics import confusion_matrix # 导入混淆矩阵函数
	from sklearn.metrics import  precision_recall_curve, roc_curve, auc
	from sklearn.preprocessing import label_binarize
	import matplotlib.pyplot as plt
	from itertools import cycle
	from sklearn import metrics
	import warnings
	warnings.filterwarnings("ignore")
	report = report_utils.Report()

	'''
	选择目标数据
	'''
	data_in = db_utils.query(conn, 'select '+ params['features'] + ',' + params['label'] + ' from ' + inputs['data_in'])

	'''
	绘制报告
	'''
	report.h1('KNN测试结果')

	'''
	拆分数据
	'''
	x_test = data_in.drop(params['label'], 1)
	y_test = data_in[params['label']]
	y_test=y_test.astype(str)
	class_names = y_test.unique()
	n_classes = len(class_names)
	y_binarize = label_binarize(y_test, classes = class_names) # 标签热编码

	'''
	模型测试
	'''
	y_fit = model.predict(x_test)  # 用模型进行预测，返回预测值
	y_score = model.predict_proba(x_test)  # 返回一个数组，数组的元素依次是X预测为各个类别的概率值
	fit_label = pd.DataFrame(y_fit, columns = [params['add_col']])

	'''
	输出预测值
	'''
	data_out = pd.concat([x_test, y_test, fit_label], axis = 1)

	'''
	报告
	'''
	report.h1('KNN算法评估脚本 ')
	cm = confusion_matrix(y_test, fit_label) # 混淆矩阵
	n_classes = len(cm)
	n_classes

	if(n_classes == 2):
	    cm = confusion_matrix(y_test, fit_label) # 混淆矩阵
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

	from sklearn.metrics import classification_report
	if(n_classes >2):
		from sklearn import preprocessing
		import numpy as np
		binarizer = preprocessing.Binarizer(threshold=0.5)
		y_score=binarizer.transform(y_score)
		target_names = class_names
		a=classification_report(y_test, fit_label,target_names=target_names)
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
	cm = confusion_matrix(y_test, fit_label) # 混淆矩阵
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

	'''
	绘制ROC曲线
	fpr：假正例率
	tpr：真正例率
	'''
	colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
	y_fit=label_binarize(y_fit, classes = class_names)
	if(n_classes == 2):
		fpr, tpr, _ = roc_curve(y_binarize.ravel(),y_fit.ravel())
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
		fpr, tpr, _ = precision_recall_curve(y_binarize.ravel(),y_fit.ravel())
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
		fpr, tpr, thresholds = metrics.roc_curve(y_binarize.ravel(),y_fit.ravel())
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
