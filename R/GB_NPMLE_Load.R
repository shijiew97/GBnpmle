#' @title
#' Loading function of Generative Bootstrap NPMLE
#'
#' @description
#' Loading a pre-trained generator for further training or Sample
#'
#' @param save_filePy the saved .pt/ckpt file directory from GB-NPMLE function.
#' @param save_fileR the saved .RData file directory from GB-NPMLE function.
#' @param option option = "Train": load a pre-trained generator and further train; option = "Sample": load a pre-trained generator and generate bootstrap samples.
#' @param gpu_ind gpu index.
#' @param N number of Monte Carlo iterations for approximating E_w.
#' @param M number of Monte Carlo iterations for approximating E_z.
#' @param num_it number of iterations for training.
#' @param verb print information while training generator.
#' @param tol tolerance to determine whether EM algorithm converges;default is 0.005.
#' @param lrDecay lrDecay = 1: using decaying learning rate.
#' @param lrpower decay rate of learning rate, default is 0.2.
#' @param boot_size the number of bootstrap samples to be generated.
#'
#' @usage
#' GB_NPMLE_Load(save_filePy=path_py, save_fileR=path_r, option="Train", gpu_ind=0,
#' N=100, M=100, num_it=2000, verb=1, tol=0.005, lrDecay=0, boot_size=NULL)
#'
#' @author
#' Shijie Wang, Minsuk Shin and Ray Bai
#'
#' @seealso \code{\link{GB_NPMLE}}, \code{\link{GB_NPMLE_Sample}}
#' @export
#'
#' @examples
#' ### Pre-training stage
#' library(reticulate)
#' set.seed(2^2+2021)
#' sigma = 0.5;n = 100
#' theta = c(rep(0,0.2*n),
#'           rep(5,0.8*n))
#' Y = theta+rnorm(n,0,sigma)
#' fit_GBnpmle = GB_NPMLE(Y=Y, param=sigma, distribution="Gaussian location",
#' num_it=200, n=n, p=1, S=100, q=100, verb=1, hidden_size=500,
#' save=1, save_path=save_path)
#'
#' ### Loading stage
#' fit_load = GB_NPMLE_Load(save_filePy=path_py, save_fileR=path_r, option="Train",
#' N=100, M=100, num_it=1800, verb=1, tol=0.005, lrDecay=0, boot_size=1000, gpu_ind=0)
#'
#' Sample_GBnpmle = GB_NPMLE_Sample(fit_load[[1]], boot_size=10000)
#' hist(Sample_GBnpmle$Theta, breaks=25, main="", xlab=expression(theta), freq=F, col="white", border=T)
#' #hist(fit_load$Theta, breaks=25, main="", xlab=expression(theta), freq=F, col="white", border=T)
#' points(x=0, y=0, pch=4, col='red', lwd=2)
#' points(x=5, y=0, pch=4, col='red', lwd=2)
#'
GB_NPMLE_Load <- function(save_filePy=NULL, save_fileR=NULL, option=NULL,
                          gpu_ind=0,  N=100, M=100, num_it=NULL, verb=1, tol=0.005,
                          lrDecay=0, lrpower=0.2, boot_size=NULL){
  require(reticulate)

  if(is.null(option) == TRUE){stop("Option is missing!")}
  if(is.null(save_filePy) == TRUE){stop("Generator file (.py) path is missing!")}
  if(is.null(save_fileR) == TRUE){stop("Generator file (.RData) is missing!")}
  if(option == "Train"){
    if(is.null(num_it) == TRUE){
      num_it = 2000
      print("Warning: Number of iterations is not specified! (Set to be 2,000)")
    }
    if(is.null(boot_size)==TRUE){boot_size=1000}
  }
  if(option == "Sample"){
    if(is.null(num_it)==TRUE){num_it=2000}
  }

  out_GBnpmle = readRDS(save_fileR)
  tau0 = out_GBnpmle[[3]]
  p = out_GBnpmle[[4]]
  l = out_GBnpmle[[5]]
  distribution = out_GBnpmle[[6]]
  s = out_GBnpmle[[7]]
  n = out_GBnpmle[[8]]
  q = out_GBnpmle[[9]]
  hidden_size = out_GBnpmle[[12]]
  param = out_GBnpmle[[13]]
  lr = out_GBnpmle[[14]]
  X = out_GBnpmle[[17]]
  Y = out_GBnpmle[[18]]

  if(is.vector(Y) == TRUE){Y = matrix(Y, ncol=1, nrow=n)}
  if(is.vector(X) == TRUE){X = matrix(X, ncol=p, nrow=n)}

  Tau0 = r_to_py(tau0, convert=FALSE)
  P = r_to_py(p, convert=FALSE)
  L = r_to_py(l, convert=FALSE)
  Dist = r_to_py(distribution, convert=FALSE)
  S = r_to_py(s, convert=FALSE)
  n1 = r_to_py(n, convert=FALSE)
  Q = r_to_py(q, convert=FALSE)
  Hidden_size = r_to_py(hidden_size, convert=FALSE)
  Param = r_to_py(param, convert=FALSE)
  Lr = r_to_py(lr, convert=FALSE)
  X = r_to_py(X, convert=FALSE)
  Y = r_to_py(Y, convert=FALSE)

  N1 = r_to_py(N, convert=FALSE)
  Num_it = r_to_py(num_it, convert=FALSE)
  Verb = r_to_py(verb, convert=FALSE)
  Tol = r_to_py(tol, convert=FALSE)
  LrDecay = r_to_py(lrDecay, convert=FALSE)
  Lrpower = r_to_py(lrpower, convert=FALSE)
  Save_file = r_to_py(save_filePy, convert = FALSE)
  Boot_size = r_to_py(boot_size, convert=FALSE)
  Option = r_to_py(option, convert=FALSE)
  Gpu_ind = r_to_py(gpu_ind, convert=FALSE)
  M = r_to_py(M, convert=FALSE)


  Load_GBnpmle <- paste(system.file(package="GBnpmle"), "GB_NPMLE_Loading.py", sep="/")
  reticulate::source_python(Load_GBnpmle)

  fit_load = GB_NPMLE_Loading(Option, Save_file, Gpu_ind, N1, M, Num_it, Verb, Tol, LrDecay, Boot_size,
                              Tau0, P, L, Dist, S, n1, Q, Hidden_size, Param, Lr, X, Y, Lrpower)

  if(option == "Sample"){
    p = as.numeric(fit_load[[3]])
    Theta_dist = matrix(unlist(fit_load[[1]], use.names=F), nrow=boot_size, ncol=p)
    out_load = list('Theta' = Theta_dist,
                    'Generation_time' = as.numeric(fit_load[[2]]))
  }

  if(option == "Train"){
    out_load = list("GBnpmle_obj" = fit_load,
                       "Neural Network structure" = list(fit_load[[1]]),
                       "EM algorithm tau" = as.numeric(fit_load[[2]]),
                       "p" = as.numeric(fit_load[[3]]),
                       "l" = as.numeric(fit_load[[4]]),
                       "distribution" = as.character(fit_load[[5]]),
                       "s" = as.numeric(fit_load[[6]]),
                       "n" = as.numeric(fit_load[[7]]),
                       "q" = as.numeric(fit_load[[8]]),
                       "Generator train time" = as.numeric(fit_load[[9]]),
                       "EM train time" = as.numeric(fit_load[[10]]),
                       "Hidden size" = as.numeric(fit_load[[11]]),
                       "Param" = as.numeric(fit_load[[12]]),
                       "lr" = as.numeric(fit_load[[13]]),
                       "N" = as.numeric(fit_load[[14]]),
                       "M" = as.numeric(fit_load[[15]]))
  }
  return(out_load)
}






















