import numpy as np
import matplotlib.pyplot as plt
import time
plt.rcParams.update({'font.size':34})
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams.update({'font.weight':'bold'})
plt.rcParams["font.family"] = "Times New Roman"
rc = {"font.family" : "serif", 
      "mathtext.fontset" : "stix"}
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
################################################################
'''
Given function
'''
def f(t,y):
	func=-np.exp(-t)
	return func
################################################################
'''
adaptH function
'''
def adaptH(p,hn,errTol,LTEn,del_y):
	if LTEn>errTol:
		rejectH=1
	else:
		rejectH=0
	fac=((errTol)/(abs(LTEn)))**(1/(p+1))
	h_recommend=0.9*hn*min(max((fac),0.3),2.0)

	return rejectH,h_recommend
################################################################
'''
Time info
'''
t0=0
tn=t0
t_end=30

time_array_BS=[]
time_array_BS=np.append(time_array_BS,tn)

time_array_Euler=[]
time_array_Euler=np.append(time_array_Euler,tn)
################################################################
'''
Initial value
'''
y0=1.0
yn=y0
y_sol=[]
y_sol=np.append(y_sol,yn)
################################################################
errTol=1e-6
################################################################
'''
time step
'''
h_int=0.2
hn=h_int
################################################################
'''
Butcher Tableau coefficients for Bogacki-Shampine Method
'''
c=np.array([0,(1/2.0),(3/4.0),1.0])
a=np.array([[0,0,0,0],
            	     [(1/2.0),0,0,0],
                     [0,(3/4.0),0,0],
                     [(2/9.0),(1/3.0), (4/9.0), 0]])

b=np.array([2/9., 1/3., 4/9., 0])
b_star=np.array([7/24., 1/4., 1/3., 1/8.])
################################################################
'''
Bogacki-Shampine algorithm starts
'''
stage=4
p=3.0   # order for Bogacki-Shampine Method
s=np.arange(1,stage,1)
k=np.zeros(stage)

start=time.time()
while tn<t_end:
	#print('-------------------------------------')
	#print('tn = ' + str(tn))
	if t_end<tn+hn:
		hn=t_end-tn
	rejectH=1
	k[0]=f(tn,yn)
	while rejectH==1:
		del_y=b[0]*k[0]
		LTEn=(b[0]-b_star[0])*k[0]
		for i0 in s:
			t_int=tn+c[i0]*hn
			sum_a_k=0
			j=i0
			for i1 in range(j):
				sum_a_k=sum_a_k+a[i0][i1]*k[i1]
			y_int=yn+hn*sum_a_k
			k[i0]=f(t_int,y_int)
			del_y=del_y+b[i0]*k[i0]
			LTEn=LTEn+(b[i0]-b_star[i0])*k[i0]
		LTEn=LTEn*hn 
		[rejectH,h_recommend]=adaptH(p,hn,errTol,LTEn,del_y)
		if rejectH==1:
			hn=h_recommend
		else:
			y_n_plus_1=yn+hn*del_y
			tn=tn+hn
			time_array_BS=np.append(time_array_BS,tn)
			y_sol=np.append(y_sol,y_n_plus_1)
			yn=y_n_plus_1
			hn=h_recommend

print('-------------------------------------')
################################################################      
end=time.time()        
execution_time=end-start
print('Bogacki-Shampine Execution time = ' + str(execution_time) + 's')
################################################################      
'''
Forward Euler
'''
tn=t0
time_Fwd_Euler=[]
time_Fwd_Euler=np.append(time_Fwd_Euler,tn)
hn=h_int

Euler_sol=[]
Euler_sol=np.append(Euler_sol,y0)
yn=y0
start=time.time()
while tn<t_end:
	#print('-------------------------------------')
	#print('tn = ' + str(tn))
	yn=Euler_sol[-1]+hn*f(tn,yn)
	tn=tn+hn
	if t_end<tn:
		break
	time_Fwd_Euler=np.append(time_Fwd_Euler,tn)
	Euler_sol=np.append(Euler_sol,yn)
end=time.time()        
execution_time=end-start
print('Forward Euler Execution time = ' + str(execution_time) + 's')
################################################################
'''
Exact solution 
'''
time_array=time_Fwd_Euler
y_exact=np.exp(-time_array)
################################################################

plt.figure(figsize=(12,8))
plt.plot(time_array_BS,y_sol,'o',linewidth=2.0,label='adaptive BS')
plt.plot(time_Fwd_Euler,Euler_sol,'k',linewidth=2.0,label='Euler')
plt.plot(time_array,y_exact,'r',linewidth=2.0,label='Exact')
plt.xticks([0,10,20,30])
plt.legend()
plt.grid()
plt.title('Intial time step = ' + str(h_int))
plt.xlabel('t')
plt.ylabel('y(t)')
plt.savefig('Solution.png', bbox_inches='tight', pad_inches=0.0)
plt.show()
