
import rpy2.robjects as ro
import pandas.rpy.common as com

ro.r("""bag.fsr<-function(x,y,B=100,gam0=.05){
# gives average coefficients from fsr.fast6.sim
ok<-complete.cases(x,y)
x<-x[ok,]                            # get rid of na's
y<-y[ok]                             # since regsubsets can't handle na's
m<-ncol(x)
n<-nrow(x)
hold<-matrix(rep(0,m*B),nrow=B)      # holds coefficients
interc<-rep(0,B)                     # holds intercepts
alphahat<-rep(0,B)                   # holds alphahats
size<-rep(0,B)                       # holds sizes
for(i in 1:B){
index<-sample(1:n,n,replace=T)
out<-fsr.fast6.sim(x=x[index,],y=y[index],gam0=gam0)
if (out$size>0) hold[i,out$x.ind]<-out$mod$coeff[2:(out$size+1)]
interc[i]<-out$mod$coeff[1]
alphahat[i]<-out$alphahat.ER
size[i]<-out$size
}                                    # ends i loop
coeff.av<-apply(hold,2, mean)
coeff.sd<-rep(0,m)
coeff.sd<-sqrt(apply(hold,2, var))
interc.av<-mean(interc)
interc.sd<-sd(interc)
amean<-mean(alphahat)
sizem<-mean(size)
prop<-rep(0,m)
for(j in 1:m){prop[j]<-sum(abs(hold[,j])>0)/B}
as.matrix(x)->x                      # in case x is a data frame
pred<-x%*%coeff.av+interc.av
return(list(coeff.av=coeff.av,coeff.sd=coeff.sd,interc.av=interc.av,pred=pred,
            interc.sd=interc.sd,prop=prop,amean=amean,sizem=sizem))
}""")

ro.r("""fsr.fast6.sim<-function(x,y,gam0=.05){
# estimated alpha for forward selection
# short output version
require(leaps)
ok<-complete.cases(x,y)
x<-x[ok,]                            # get rid of na's
y<-y[ok]                             # since regsubsets can't handle na's
m<-ncol(x)
n<-nrow(x)
if(m >= n) m1 <- n-5  else m1<-m     # to get rid of NA's in pv
vm<-1:m1
as.matrix(x)->x                      # in case x is a data frame
pvm<-rep(0,m1)                       # to create pvm below
regsubsets(x,y,method="forward")->out.x
pv.orig<-1-pf((out.x$rss[vm]-out.x$rss[vm+1])*(out.x$nn-(vm+1))/out.x$rss[vm+1],1,out.x$nn-(vm+1))
for (i in 1:m1){pvm[i]<-max(pv.orig[1:i])}  # sequential max of pvalues
alpha<-c(0,pvm)
ng<-length(alpha)
S<-rep(0,ng)                         # will contain num. of true entering in orig.
real.seq<-data.frame(var=(out.x$vorder-1)[2:(m1+1)],pval=pv.orig,
         pvmax=pvm,Rsq=round(1-out.x$rss[2:(m1+1)]/out.x$rss[1],4))
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
ind<-(ghat <= gam0 & alpha<=alphamax)*1
Sind<-S[max(which(ind > 0))]          # model size with ghat just below gam0
alphahat.fast<-(1+Sind)*gam0/(m-Sind)  # ER est.
size1<-sum(pvm<=alphahat.fast)+1       # size of model including intercept
colnames(x)<-colnames(x,do.NULL=F,prefix="")      # corrects for no colnames
x<-x[,colnames(x)[(out.x$vorder-1)[2:size1]]]
if(size1>1) x.ind<-(out.x$vorder-1)[2:size1]  else x.ind<-0
if (size1==1) {mod <- lm(y~1)} else {mod <- lm(y~x)}
return(list(mod=mod,size=size1-1,x.ind=x.ind,alphahat.ER=alphahat.fast))
}""")

ro.r('ncaa <- as.matrix(read.table("ncaa_data2.txt",header=T))')

print "NCSU R Results:\n"

print ro.r('system.time(bag.fsr(x=ncaa[,1:19],y=ncaa[,20],B=200)->out.ncaa)')

print ro.r('paste("Mean of estimated alpha-to-enter:",round(out.ncaa$amean,4))')
print ro.r('paste("Mean size of selected model:",round(out.ncaa$sizem,4))')