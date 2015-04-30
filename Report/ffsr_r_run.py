
import rpy2.robjects as ro
import pandas.rpy.common as com

ro.r("""fsr.fast<-function(x,y,gam0=.05,digits=4,print=T,plot=F){
# estimated alpha for forward selection using Fast FSR (no simulation)
# typical call: fsr.fast(x=ncaa2[,1:19],y=ncaa2[,20])->out
# for use inside simulation loops, set print=F and plot=F
# version 7 circa Nov. 2009, modified to handle partially blank colnames
require(leaps)
ok<-complete.cases(x,y)
x<-x[ok,]                            # get rid of na's
y<-y[ok]                             # since regsubsets can't handle na's
m<-ncol(x)
n<-nrow(x)
if(m >= n) m1 <- n-5  else m1<-m     # to get rid of NA's in pv
vm<-1:m1
as.matrix(x)->x                      # in case x is a data frame
if(any(colnames(x)==""))colnames(x)<-NULL       # if only partially named columns
colnames(x)<-colnames(x,do.NULL=F,prefix="")    # corrects for no colnames
pvm<-rep(0,m1)                       # to create pvm below
regsubsets(x,y,method="forward")->out.x
pv.orig<-1-pf((out.x$rss[vm]-out.x$rss[vm+1])*(out.x$nn-(vm+1))/out.x$rss[vm+1],1,out.x$nn-(vm+1))
for (i in 1:m1){pvm[i]<-max(pv.orig[1:i])}  # sequential max of pvalues
alpha<-c(0,pvm)
ng<-length(alpha)
S<-rep(0,ng)                         # will contain num. of true entering in orig.
real.seq<-data.frame(var=(out.x$vorder-1)[2:(m1+1)],pval=pv.orig,
         pvmax=pvm)#,Rsq=round(1-out.x$rss[2:(m1+1)]/out.x$rss[1],4))
for (ia in 2:ng){                    # loop through alpha values for S=size
S[ia] <- sum(pvm<=alpha[ia])         # size of models at alpha[ia], S[1]=0
}
ghat<-(m-S)*alpha/(1+S)              # gammahat_ER
# add additional points to make jumps
alpha2<-alpha[2:ng]-.0000001
ghat2<-(m-S[1:(ng-1)])*alpha2/(1+S[1:(ng-1)])
zp<-data.frame(a=c(alpha,alpha2),g=c(ghat,ghat2))
zp<-zp[order(zp$a),]
gmax<-max(zp$g)
index.max<-which.max(zp$g)           # index of largest ghat
alphamax<-zp$a[index.max]            # alpha with largest ghat
# gmax<-max(ghat)
# index.max<-which.max(ghat)           # index of largest ghat
# alphamax<-alpha[index.max]           # alpha with largest ghat
ind<-(ghat <= gam0 & alpha<=alphamax)*1
Sind<-S[max(which(ind > 0))]           # model size with ghat just below gam0
alphahat.fast<-(1+Sind)*gam0/(m-Sind)  # ER est.
size1<-sum(pvm<=alphahat.fast)+1       # size of model including intercept
x<-x[,colnames(x)[(out.x$vorder-1)[2:size1]]]
if(size1>1) x.ind<-(out.x$vorder-1)[2:size1]  else x.ind<-0
if (size1==1) {mod <- lm(y~1)} else {mod <- lm(y~x)}
# ghat3<-(m-size1+1)*alpha/(1+S)         # uses final ku est.
ghat4<-(m-size1+1)*alpha/(1+0:m)
#res<-data.frame(real.seq,ghigh=ghat2,glow=ghat[2:ng])
alphas<-gam0 * (1. + S[2:ng]) / (m - S[2:ng])
res<-data.frame(S=S[2:ng],real.seq,alpha=alphas,g=ghat[2:ng])
if(print)print(round(res,digits))
#if(plot){
#plot(zp$a,zp$g,type="b",xlab="Alpha",ylab="Estimated Gamma",xlim=c(0,alphamax))
#points(alphahat.fast,gam0,pch=19)
#lines(c(-1,alphahat.fast),c(gam0,gam0))
#lines(c(alphahat.fast,alphahat.fast),c(-1,gam0))
#}  # ends plot
return(list(res=round(res,digits),mod=mod,size=size1-1,x.ind=x.ind,alphahat.ER=alphahat.fast))
}""")

ro.r('ncaa <- as.matrix(read.table("ncaa_data2.txt",header=T))')

print "NCSU R Results:\n"

print ro.r('system.time(fsr.fast(x=ncaa[,1:19],y=ncaa[,20]))')