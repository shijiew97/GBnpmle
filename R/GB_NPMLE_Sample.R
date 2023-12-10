#' @title
#' Sampling function of Generative Bootstrap NPMLE
#'
#' @description
#' Generate bootstrap samples from a trained NPMLE generator
#'
#' @param fit_GBnpmle a GB_NPMLE object evaluated by "GB_NPMLE" function.
#' @param boot_size the number of bootstrap samples to be generated.
#'
#' @usage
#' GB_NPMLE_Sample(fit_GBnpmle, boot_size=10000)
#' @author
#' Shijie Wang, Minsuk Shin and Ray Bai
#' @seealso \code{\link{GB_NPMLE}}, \code{\link{GB_NPMLE_Load}}
#' @export
#' @examples
#' #sample_GBnpmle = GB_NPMLE_Sample(fit_GBnpmle, boot_size=10000)
GB_NPMLE_Sample <- function(fit_GBnpmle, boot_size){

  if(is.null(boot_size)==TRUE){
    boot_size = 10000
    print("Warning: The bootstrap sample size is un-specified and set to 10000 instead")
  }
  Boot_size = r_to_py(boot_size, convert=FALSE)

  Sample_GBnpmle <- paste(system.file(package="GBnpmle"), "GB_NPMLE_Sampling.py", sep="/")
  reticulate::source_python(Sample_GBnpmle)

  fit_sample = GB_NPMLE_Sampling(fit_GBnpmle, Boot_size)

  p = as.numeric(fit_sample[[3]])
  Theta_dist = matrix(unlist(fit_sample[[1]], use.names=F), nrow=boot_size, ncol=p)

  out_sample = list('Theta' = Theta_dist,
                    'Generation_time' = as.numeric(fit_sample[[2]]))
  return(out_sample)
}
