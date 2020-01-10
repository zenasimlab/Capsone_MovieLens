## Hui Xin Sim
## MovieLens Project 
## HarvardX: PH125.9x - Capstone Project
## https://github.com/zenasimlab

#############################################
# MovieLens Rating Prediction Project Code
##############################################

###Introduction###

## Dataset ##

################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")


dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)



# Validation dataset can be further modified by removing rating column
validation_CM <- validation  
validation <- validation %>% select(-rating)

#Extra libraries that might be useful
library(ggplot2)
library(lubridate)

## Methods and Analysis ##
## Data Analysis ##

# The presentation of 1st few rows of edx:
head(edx)

# Summary Statistics of edx
summary(edx)

# total number of observations
tot_observation <- length(edx$rating) + length(validation$rating)


# Modify the year as a column in the edx & validation datasets
edx <- edx %>% mutate(year = as.numeric(str_sub(title,-5,-2)))
validation <- validation %>% mutate(year = as.numeric(str_sub(title,-5,-2)))
validation_CM <- validation_CM %>% mutate(year = as.numeric(str_sub(title,-5,-2)))

# Modify the genres variable in the edx & validation dataset (column separated)
split_edx  <- edx  %>% separate_rows(genres, sep = "\\|")
split_valid <- validation   %>% separate_rows(genres, sep = "\\|")
split_valid_CM <- validation_CM  %>% separate_rows(genres, sep = "\\|")

# Number of unique movies and users in the edx dataset
edx%>%
  summarize(n_users=n_distinct(userId),n_movies=n_distinct(movieId))


# Top 10 movies ranked in order of the number of rating
edx%>%
  group_by(movieId,title)%>%
  summarize(count=n())%>%
  arrange(desc(count))


# Ratings distribution

vec_ratings<-as.vector(edx$rating)
unique(vec_ratings)

vect_ratings<-vec_ratings[vec_ratings!=0]
vec_ratings<-factor(vec_ratings)
qplot(vec_ratings)+ggtitle("Ratings Distribution")

# The distribution of each user's ratings for movie
edx%>%count(userId)%>%
  ggplot(aes(n))+
  geom_histogram(bins=30,color="black")+
  scale_x_log10()+
  ggtitle("Users")


# Plot number of ratings per movie
edx%>%count(movieId)%>%
  ggplot(aes(n))+
  geom_histogram(bins=30,color="black")+
  scale_x_log10()+
  xlab("Number of ratings")+
  ylab("Number of movies")+
  ggtitle("Number of ratings per movie")
  

# Plot number of ratings given by users
edx%>%count(userId)%>%
  ggplot(aes(n))+
  geom_histogram(bins=30,color="black")+
  scale_x_log10()+
  xlab("Number of ratings")+
  ylab("Number of users")+
  ggtitle("Number of ratings given by users")
  

# Rating vs release year

edx%>%group_by(year)%>%
  summarize(rating=mean(rating))%>%
  ggplot(aes(year,rating))+
  geom_point()+
  geom_smooth()


### Modelling Approach and Results ###


# Compute the dataset's mean rating
mu<-mean(edx$rating)
mu

rmse_naive<-RMSE(validation_CM$rating,mu)
rmse_naive

## Save Results in Data Frame
rmse_results<-data_frame(method="Naive Analysis by Mean",RMSE=rmse_naive)
rmse_results%>%knitr::kable()


## Movie Effects model ##

#Penalty Term(b_i)-Movie Effect

movie_avgs_norm<-edx %>%
  group_by(movieId)%>%
  summarize(b_i=mean(rating-mu))
movie_avgs%>%qplot(b_i,geom ="histogram",bins = 20,data= .,color=I("black"))


predicted_ratings_movie_norm<-validation%>%
  left_join(movie_avgs_norm,by='movieId')%>%
  mutate(pred=mu+b_i)
model_1_rmse<-RMSE(validation_CM$rating,predicted_ratings_movie_norm$pred)
rmse_results<-bind_rows(rmse_results,
                        data_frame(method="Movie Effect Model",
                                   RMSE=model_1_rmse))

# save rmse results in a table
rmse_results%>%knitr::kable()
rmse_results


## Movie and User Effects Model ##

# Penalty Term(b_u)-User Effect
user_avgs_norm <- edx %>% 
  left_join(movie_avgs_norm, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))
user_avgs_norm %>% qplot(b_u, geom ="histogram", bins = 30, data = ., color = I("black"))

# User test set,join movie averages and user averages
# Prediction equals the mean with user effect b_u & movie effect b_i
predicted_ratings_user_norm<-validation%>%
  left_join(movie_avgs_norm,by='movieId')%>%
  left_join(user_avgs_norm,nu='userId')%>%
  mutate(pred=mu+b_i+b_u)

# test and save rmse results
model_2_rmse<-RMSE(validation_CM$rating,predicted_ratings_user_norm$pred)
rmse_results<-bind_rows(rmse_results,
                        data_frame(method="Movie and User Effect Model",
                                   RMSE=model_2_rmse))
rmse_results%>%knitr::kable()


## Regularized movie and user effect model ##

# lambda is a tuning parameter
# User cross-validation to choose it
lambdas<-seq(0,10,0.25)

# For each lambda,find b_i & b_u, followed by rating prediction & testing
rmses<-sapply(lambdas,function(l){
  
  mu<-mean(edx$rating)
  
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))

  b_u <- edx %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+l))

 predicted_ratings <- validation %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

return(RMSE(validation_CM$rating,predicted_ratings))
})

# Plot rmses vs lamdas to select the optimal lambda
qplot(lambdas,rmses)

# The optimal lamda
lambda<-lambdas[which.min(rmses)]
lambda

# Compute regularized estimates of b_i using lambda
movie_avgs_reg<-edx %>% 
  group_by(movieId)%>% 
  summarize(b_i=sum(rating-mu)/(n()+lambda),n_i = n())

# Compute regularized estimates of b_u using lambda
user_avgs_reg<-edx %>% 
  left_join(movie_avgs_reg,by='movieId')%>%
  group_by(userId)%>%
  summarize(b_u=sum(rating-mu-b_i)/(n()+lambda),n_u = n())

# Predict ratings
predicted_ratings_reg<-validation%>% 
  left_join(movie_avgs_reg,by='movieId')%>%
  left_join(user_avgs_reg,by='userId')%>%
  mutate(pred=mu+b_i+b_u) %>% 
  .$pred

# Test and save results

model_3_rmse<-RMSE(validation_CM$rating,predicted_ratings_reg)
rmse_results<-bind_rows(rmse_results,
                        data_frame(method="Regularized Movie and User Effect Model",  
                                   RMSE=model_3_rmse))

rmse_results%>%knitr::kable()
rmse_results


## Results ##

# RMSE results overview
rmse_results%>%knitr::kable()