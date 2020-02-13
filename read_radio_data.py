import sjoert
import glob
import datetime
import astropy.time
import astropy.io.ascii

ami = astropy.io.ascii.read('./data/at2019dsg_AMI.csv', delimiter=',')

data_rec = np.zeros(100, dtype=[('date','U13'),('mjd', float), ('nu_GHz',float), ('inst','U10'), ('flux_mJy', float), ('eflux_mJy', float), ])
l=0 
for a in ami:
	dstr1 = a['Start Date'] 
	dl1 = np.array(dstr1.replace('/', ' ').replace(':', ' ').split(' '), dtype=np.int)
	mjd1 =  sjoert.simtime.datetomjd(datetime.datetime(dl1[2],dl1[0],dl1[1], dl1[3],dl1[4],dl1[5]))  
	dstr2 = a['End Date'] 
	dl2 = np.array(dstr2.replace('/', ' ').replace(':', ' ').split(' '), dtype=np.int)
	mjd2 =  sjoert.simtime.datetomjd(datetime.datetime(dl2[2],dl2[0],dl2[1], dl2[3],dl2[4],dl2[5]))  
	print (mjd1, mjd2)
	
	data_rec['mjd'][l] 		= str((mjd1+mjd2)/2.)[0:10]
	data_rec['flux_mJy'][l] 	= a['Peak'] / 1e3
	data_rec['eflux_mJy'][l]	= float(str(a['Peak Error'])[0:6]) / 1e3
	data_rec['date'][l] 	= sjoert.simtime.mjdtodate(data_rec['mjd'][l]).strftime('%y/%m/%d')    
	data_rec['inst'][l] 	= 'AMI'
	data_rec['nu_GHz'][l] 		= 15.5
	l+=1


emerlin = astropy.io.ascii.read('./data/at2019dsg_eMERLIN.dat')
for m in emerlin:
	data_rec['mjd'][l] 	= m['mjd']
	data_rec['flux_mJy'][l] = m['flux_mJy']
	data_rec['eflux_mJy'][l] = m['flux_err_mJy']
	data_rec['date'][l] = sjoert.simtime.mjdtodate(data_rec['mjd'][l]).strftime('%y/%m/%d')    
	data_rec['inst'][l] = 'eMERLIN'
	data_rec['nu_GHz'][l] = 5.07
	l+=1	

meerkat = astropy.io.ascii.read('./data/at2019dsg_MeerKAT.txt')
for m in meerkat:
	data_rec['mjd'][l] 	= m['mjd']
	data_rec['flux_mJy'][l] = m['flux_mJy']
	data_rec['eflux_mJy'][l] = m['flux_err_mJy']
	data_rec['date'][l] = sjoert.simtime.mjdtodate(data_rec['mjd'][l]).strftime('%y/%m/%d')    
	data_rec['inst'][l] = 'MeerKAT'
	data_rec['nu_GHz'][l] = 1.4
	l+=1	

vlalist = glob.glob('./data/at2019dsg_20*')
for vlal in vlalist:
	vla = astropy.io.ascii.read(vlal)
	for v in vla:
		dstr = vlal.split('_')[-1].split('.')[0] 
		data_rec['mjd'][l] 		= sjoert.simtime.datetomjd(datetime.datetime(int(dstr[0:4]),int(dstr[4:6]),int(dstr[6:])))  
		data_rec['flux_mJy'][l] 	= v['col2']	
		data_rec['eflux_mJy'][l]	= v['col3']		
		dstrout = sjoert.simtime.mjdtodate(data_rec['mjd'][l]).strftime('%y/%m/%d')    	
		data_rec['date'][l] 	= dstrout
		data_rec['inst'][l] 	= 'VLA'
		data_rec['nu_GHz'][l] 		= v['col1']
		l+=1
		print (dstr, dstrout)



data_rec = data_rec[0:l]

data_rec = data_rec[np.argsort(data_rec['mjd'])]
astropy.io.ascii.write(data_rec, './data/at2019dsg_merged.dat', overwrite=True, format='fixed_width')

plt.clf()
mjd_plot = [58034, 58625,58637, 58653, 58703, 58761, 58818]

for i, mjd in enumerate(mjd_plot): 
	it = abs(data_rec['mjd']-mjd)<10
	ii = it * (data_rec['eflux_mJy']>0)
	print (sum(ii))
	lbl = sjoert.simtime.mjdtodate(np.mean(data_rec['mjd'][it])).strftime('%y/%m/%d')
	line = plt.errorbar(data_rec[ii]['nu_GHz'], data_rec[ii]['flux_mJy'], data_rec[ii]['eflux_mJy'], fmt='o', label=lbl, zorder=10-i, alpha=0.8)
	ii = it * (data_rec['eflux_mJy']<0)
	plt.errorbar(data_rec[ii]['nu_GHz'], data_rec[ii]['flux_mJy'], fmt='v', zorder=10-i, alpha=0.8, color=line[0].get_color())


plt.xlabel('Frequency (Ghz)')
plt.ylabel('Flux (mJy)')
plt.xlim(1, 20)
plt.legend()
plt.savefig('./data/at2019dsg_radiomerged.pdf')


