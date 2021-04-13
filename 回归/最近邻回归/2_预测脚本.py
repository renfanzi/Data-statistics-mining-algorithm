# <editable>
# 在此处添加算法描述
# </editable>
# conn：             数据库连接
# model：            模型
# inputs:            输入集合
# params:            参数
# outputs:           输出集合
# reportFileName：   预测中所生成的报告文件的存储路径
def doPredict(conn, model, inputs, params, outputs, reportFileName):
    #<editable>
	'''
	选择目标数据
	'''
	data_in = db_utils.query(conn, 'select '+ params['columns'] + ' from ' + inputs['data_in'])

	'''
	模型预测
	'''
	y_fit = model.predict(data_in)  # 用模型进行预测，返回预测值
	fit_label = pd.DataFrame(y_fit, columns = [params['add_col']])

	'''
	输出预测值
	'''
	data_out = pd.concat([data_in, fit_label], axis = 1)

	'''
	将结果写出
	'''
	db_utils.dbWriteTable(conn, outputs['data_out'], data_out)
    #</editable>
