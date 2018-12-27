import sys

s = 'java 调用有第三方库的python脚本成功'
print(s)
print(sys.argv[1].encode('GBK').decode('utf-8'))