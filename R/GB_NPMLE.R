#' @title
#' Generative Bootstrap NPMLE
#'
#' @description
#' Train the generator of GB-NPMLE (Generative Bootstrap NPMLE).GB-NPMLE
#' mainly focuses on NPMLE estimation of univariate mixture model estimation.
#' python(>=3.7) and pytorch
#' are needed to be installed in advance. R pakcage 'reticulate' is also required.
#'
#' @param X predictor Variable.
#' @param Y response Variable.
#' @param distribution probabilistic distribution of Y, including:
#' "Gaussian location", "Poisson", "Gamma rate" and "Binomial".
#' @param param nuisance parameter value. Param = 0 indicates there is no nuisance parameter.
#' @param S size of subgroup. Sample size/S returns total number of subgroups.
#' @param hidden_size number of hidden neurons at each layer.
#' @param num_it number of iterations for training.
#' @param l number of replications of generator's output.
#' @param lr0 learning rate, default is 0.0005.
#' @param lrDecay lrDecay = 1: using decaying learning rate.
#' @param lrpower decay rate of learning rate, default is 0.2.
#' @param gpu_ind gpu index.
#' @param M number of Monte Carlo iterations for approximating E_z.
#' @param q length of auxiliary random variable z.
#' @param N number of Monte Carlo iterations for approximating E_w.
#' @param verb print information while training generator.
#' @param tol tolerance to determine whether EM algorithm converges;default is 0.005.
#' @param p the dimension of Predictor Variable;Default is 1.
#' @param save save the trained generator function. 1: save; 0: no save. Default is 0.
#' @param save_path the directory path where the trained generator function will be saved.
#' @usage
#' GB_NPMLE(X=NULL, Y, param=0, distribution=NULL, S=100, p=1,
#' hidden_size=500, save=0, num_it=NULL, q=100, M=100, zm=150,
#' gpu_ind=0, N=100, lr0=0.005, verb=1, tol=0.005, n=NULL, lrDecay = 0,
#' save_path=NULL)
#'
#' @return
#' GB_NPMLE function returns a list which involves NPMLE bootstrap samples, Generator training time, EM algorithm convergence time and Bootstrap sample generation time.
#'
#' @author
#' Shijie Wang, Minsuk Shin and Ray Bai
#' @seealso \code{\link{GB_NPMLE_Sample}},\code{\link{GB_NPMLE_Load}}
#' @export
#' @examples
#' ### Gaussian Location Mixture model example
#' library(reticulate)
#' set.seed(2^2+2021)
#' sigma = 0.5;n = 100
#' theta = c(rep(0,0.2*n),
#'           rep(5,0.8*n))
#' Y = theta+rnorm(n,0,sigma)
#' fit_GBnpmle = GB_NPMLE(Y=Y, param=sigma, distribution="Gaussian location", num_it=1000, n=n, p=1, S=100, q=100, verb=1, hidden_size=500)
#' Sample_GBnpmle = GB_NPMLE_Sample(fit_GBnpmle[[1]], boot_size=10000)
#' hist(Sample_GBnpmle$Theta, breaks=25, main="", xlab=expression(theta), freq=F, col="white", border=T)
#' points(x=0, y=0, pch=4, col='red', lwd=2)
#' points(x=5, y=0, pch=4, col='red', lwd=2)
#' ### Gamma shape Mixture model example
#' set.seed(1^2+2021)
#' gamma_shape = 10;n = 100
#' theta = c(rep(1,0.2*n),
#'           rep(10,0.8*n))
#' Y = rgamma(n,rate=theta,shape=gamma_shape)
#' fit_GBnpmle = GB_NPMLE(Y=Y, param=gamma_shape, distribution="Gamma rate", num_it=2000, n=n, p=1, S=100, q=100, verb=1, hidden_size=500)
#' Sample_GBnpmle = GB_NPMLE_Sample(fit_GBnpmle[[1]], boot_size=10000)
#' hist(Sample_GBnpmle$Theta, breaks=25, main="", xlab=expression(theta), freq=F, col="white", border=T, xlim=c(0,14))
#' points(x=1, y=0, pch=4, col='red', lwd=2)
#' points(x=10, y=0, pch=4, col='red', lwd=2)
#' ### Binomial probability Mixture model example
#' set.seed(1^2+2021)
#' bino_n = 10;n = 100
#' theta = c(rep(0.2,0.5*n),
#'           rep(0.8,0.5*n))
#' Y = rbinom(n, size=bino_n, prob=theta)
#' fit_GBnpmle = GB_NPMLE(Y=Y, param=bino_n, distribution="Binomial", num_it=2000, n=n, p=1, S=100, q=100, verb=1, hidden_size=500)
#' Sample_GBnpmle = GB_NPMLE_Sample(fit_GBnpmle[[1]], boot_size=10000)
#' hist(Sample_GBnpmle$Theta, breaks=25, main="", xlab=expression(theta), freq=F, col="white", border=T)
#' points(x=0.2, y=0, pch=4, col='red', lwd=2)
#' points(x=0.8, y=0, pch=4, col='red', lwd=2)
#' ### Posisson Real data analysis
#' #Thailand datasaet
#' #Y = c(rep(0,120),rep(1,64),rep(2,69),rep(3,72),rep(4,54),rep(5,35),rep(6,36),rep(7,25),
#' #rep(8,25),rep(9,19),rep(10,18),rep(11,18),rep(12,13),rep(13,4),rep(14,3),rep(15,6),
#' #rep(16,6),rep(17,5),rep(18,1),rep(19,3),rep(20,1),rep(21,2),rep(23,1),rep(24,2))
#' #Mortality dataset
#' Y = c(rep(0,162),rep(1,267),rep(2,271),rep(3,185),rep(4,111),rep(5,61),rep(6,27),rep(7,8),
#' rep(8,3),rep(9,1))
#' n = length(Y)
#' fit_GBnpmle = GB_NPMLE(Y=Y, param=0, distribution="Poisson", boot_size=1000, num_it=2000, lrDecay=0, n=n, p=1, S=100, q=100, verb=1, hidden_size=500)
#' Sample_GBnpmle = GB_NPMLE_Sample(fit_GBnpmle[[1]], boot_size=10000)
#' hist(Sample_GBnpmle$Theta, breaks=25, main="", xlab=expression(theta), freq=F, col="white", border=T)
GB_NPMLE <- function(X=NULL, Y, param=0, distribution=NULL, S=100, hidden_size=500,
                     save=0, num_it=NULL, l=100, M=100, q=150, p=1,
                     gpu_ind=0, N=100, lr0=0.0005, verb=1, tol=0.005,
                     n=NULL, lrDecay=0, lrpower=0.2, save_path=NULL){
  require(reticulate)

  if(is.null(n) == TRUE){stop("Sample size n is missing!")}
  if(is.null(X) == TRUE){X = matrix(1, ncol=p, nrow=n)}
  if(is.vector(Y) == TRUE){Y = matrix(Y, ncol=1, nrow=n)}
  if(is.vector(X) == TRUE){X = matrix(X, ncol=p, nrow=n)}

  if(is.null(dist) == TRUE){stop("Distribution of Y is missing!")}
  if(is.null(num_it) == TRUE){
    num_it = 2000
    print("Warning: Number of iterations is not specified! (Set to be 2,000)")}

  today = Sys.Date()
  if(is.null(save_path)==TRUE){path = ""}
  save_path_py = paste(save_path, "G_", today, ".pt", sep="")

  Have_torch = reticulate::py_module_available("torch")
  Have_pandas = reticulate::py_module_available("pandas")
  Have_numpy = reticulate::py_module_available("numpy")
  Have_time = reticulate::py_module_available("time")
  if (!Have_torch) stop("Pytorch is not installed!")
  if (!Have_pandas) stop("Pandas is not installed!")
  if (!Have_numpy) stop("Numpy is not installed!")
  if (!Have_time) stop("Time is not installed!")

  Y = r_to_py(Y, convert=FALSE)
  X = r_to_py(X, convert=FALSE)
  Num_it = r_to_py(num_it, convert=FALSE)
  S = r_to_py(S, convert=FALSE)
  Dist = r_to_py(distribution, convert=FALSE)
  Save = r_to_py(save, convert=FALSE)
  Q = r_to_py(q, convert=FALSE)
  M = r_to_py(M, convert=FALSE)
  L = r_to_py(l, convert=FALSE)
  Param = r_to_py(param, convert=FALSE)
  N1 = r_to_py(N, convert=FALSE)
  Hidden_size = r_to_py(hidden_size, convert=FALSE)
  n1 = r_to_py(n, convert=FALSE)
  P = r_to_py(p, convert=FALSE)
  lr0 = as.numeric(lr0)
  Lr0 = r_to_py(lr0, convert=FALSE)
  Gpu_ind = r_to_py(gpu_ind, convert=FALSE)
  Verb = r_to_py(verb, convert=FALSE)
  Tol = r_to_py(tol, convert=FALSE)
  LrDecay = r_to_py(lrDecay, convert=FALSE)
  Lrpower = r_to_py(lrpower, convert=FALSE)
  Save_path = r_to_py(save_path_py, convert = FALSE)

  Code_GBnpmle <- paste(system.file(package="GBnpmle"), "GB_NPMLE_function.py", sep="/")
  reticulate::source_python(Code_GBnpmle)

  fit_GBnpmle = GB_NPMLE_train(X, Y, S, Hidden_size, Num_it, L, M, Q, Gpu_ind, N1, n1, P, Dist,
                         Param, Lr0, Verb, Tol, LrDecay, Lrpower, Save, Save_path)

  out_GBnpmle = list("GBnpmle_obj" = fit_GBnpmle,
                     "Neural Network structure" = list(fit_GBnpmle[[1]]),
                     "EM algorithm tau" = as.numeric(fit_GBnpmle[[2]]),
                     "p" = as.numeric(fit_GBnpmle[[3]]),
                     "l" = as.numeric(fit_GBnpmle[[4]]),
                     "distribution" = as.character(fit_GBnpmle[[5]]),
                     "s" = as.numeric(fit_GBnpmle[[6]]),
                     "n" = as.numeric(fit_GBnpmle[[7]]),
                     "q" = as.numeric(fit_GBnpmle[[8]]),
                     "Generator train time" = as.numeric(fit_GBnpmle[[9]]),
                     "EM train time" = as.numeric(fit_GBnpmle[[10]]),
                     "Hidden size" = as.numeric(fit_GBnpmle[[11]]),
                     "Param" = as.numeric(fit_GBnpmle[[12]]),
                     "lr" = as.numeric(fit_GBnpmle[[13]]),
                     "N" = as.numeric(fit_GBnpmle[[14]]),
                     "M" = as.numeric(fit_GBnpmle[[15]]),
                     "X" = as.numeric(fit_GBnpmle[[16]]),
                     "Y" = as.numeric(fit_GBnpmle[[17]]))
#  NN, tau, p, k, dist, s, n, zm, Generator_time, EM_time, hidden_size, param, lr, N, M, X, Y, tol, lrDecay

  if(save == 1){
    save_path_r = paste(save_path, "GBnpmle_obj_", today, ".RData", sep="")
    saveRDS(out_GBnpmle, save_path_r)
  }

  return(out_GBnpmle)
  }
