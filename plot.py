from numpy import loadtxt

agent_name = 'a2c'
filename=input(agent_name + '_stat')

with open(filename,'rt',encoding='UTF-8')as raw_data:

    data=loadtxt(raw_data,delimiter=',')

    print(data)