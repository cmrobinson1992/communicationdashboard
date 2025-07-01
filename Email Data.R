# Loading Data
library(tidyverse)
library(tidytext)
library(hunspell)
library(purrr)
library(reticulate)
# source_python("emails.py")
# source_python("emails eda.py")
readRenviron("credentials.renviron")
# categorized Y
# removed duplicates
# removed illegible emails 


# Datapath <- setwd(Sys.getenv("datafile"))
emaildata <- read.csv('./Data/Email Data Run 03.10.2025.csv')
emaildata$Subfolder <- gsub("Inbox/", "", emaildata$Subfolder)
emaildata <- emaildata |>
  mutate(Direction = ifelse(Direction == 'Incoming', 'Incoming', 'Outgoing'))
# Filter out Drafts
# NEED: Start Date, User, User Color, Category, End Date, Subject, Body, Attachments?, Incoming/Outgoing, EmailChainId, Category (later)
incoming <- emaildata |> dplyr::filter(Direction == 'Incoming')
outgoing <- emaildata |> dplyr::filter(Direction != 'Incoming')
emaildata$Type <- 'Email'
emaildata$Direction <- as.factor(emaildata$Direction)
emaildata$ChainID <- as.factor(emaildata$ChainID)
emaildata$DateReceived <- as.POSIXct(sub(" .*$", "", emaildata$DateReceived), format = '%m/%d/%Y')
emaildata$EmailID <- row_number(emaildata)
emaildata <- emaildata %>%
  separate_wider_delim(Category, ",", names_sep = "_", too_few = "align_start") %>%
  pivot_longer(cols = starts_with("Category_"), names_to = "Category_Num", values_to = "Category", values_drop_na = TRUE) %>%
  filter(!is.na('Category')) %>%
  select(-Category_Num)

# unique_categories <- emaildata$Category %>%
#  str_split(",\\s*") %>%
#  unlist() %>%
#  unique()

# for (cat in unique_categories) {
#  emaildata[[cat]] <- as.integer(str_detect(emaildata$Category, fixed(cat)))
# }
emaildata$Month <- format(emaildata$DateReceived, '%m')
emaildata$Year <- format(emaildata$DateReceived, '%Y')
# EDA
summary_data <- emaildata |>
  group_by(Month, Direction) |>
  summarise(Count = n(), .groups = "drop")

mean_counts <- summary_data |>
  group_by(Direction) |>
  summarise(mean_count = mean(Count))

x_label_pos <- length(unique(summary_data$Month)) + 1.2  # Move labels further right
y_max <- max(summary_data$Count) * 1.1
ggplot(summary_data, aes(x = Month, y = Count, fill = Direction)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.7), width = 0.6) +
  geom_hline(data = mean_counts, aes(yintercept = mean_count, color = Direction), linetype = "dashed", size = 1) +
  geom_text(data = mean_counts, aes(x = x_label_pos, y = mean_count, label = round(mean_count, 1), color = Direction),
            hjust = -0.2, vjust = 0.5, size = 10, fontface = "bold") +
  scale_fill_manual(values = c('Incoming' = '#009eed', 'Outgoing' = '#EC8000')) +
  scale_color_manual(values = c('Incoming' = '#009eed', 'Outgoing' = '#EC8000')) +
  labs(title = "Email Count by Month and Direction",
       x = "Month",
       y = "Count",
       fill = "Direction") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 22), 
        plot.title = element_text(hjust = 0.5, size = 32),
        axis.title = element_text(size = 24),
        axis.text.y = element_text(size = 22),
        legend.position = c(1.01, 0.2),       
        legend.justification = c(0, 0),   
        legend.box = "vertical",
        plot.margin = margin(10, 100, 20, 10)) + 
  coord_cartesian(clip = "off")


# Count by User
## Ariel = 0099ff, Nathan = c6e0b4, Cathy = 00b050, Shantell = 00b050, Tyler = C00000, Porschea = cc0099, Robyn = 817300, Ruben = 92d050, Jennifer = 8497b0, kp = ffd966
Usercolors <- emaildata |>
  mutate(UserColor = ifelse(Category == "Ariel", "#0099ff", ifelse(Category == "Nathan", "#c6e0b4", ifelse(Category == "Cathy", "#00b050",
                      ifelse(Category == "Shantell", "#015025", ifelse(Category == "Tyler", "#C00000", ifelse(Category == "Porschea", "#cc0099",
                      ifelse(Category == "Robyn", "#817300", ifelse(Category == "Ruben", "#92d050", ifelse(Category == "Jennifer", "#8497b0", "")))))))))) |>
  select(User = Category, UserColor) |>
  filter(!(UserColor == "")) |>
  distinct(.keep_all = TRUE) 

UserSummary <- emaildata|>
  mutate(User = Category) |>
  filter(!is.na(User)) |>
  group_by(Month, User) |>
  summarise(Count = n()) |>
  filter(User %in% Usercolors$User)

mean_user_counts <- UserSummary |>
  group_by(User) |>
  summarise(mean_count = mean(Count))

x_label_pos <- length(unique(summary_data$Month)) + 1.2  # Move labels further right
y_max <- max(UserSummary$Count) * 1.1
color_vector = setNames(Usercolors$UserColor, Usercolors$User)
ggplot(UserSummary, aes(x = Month, y = Count, fill = User)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.7), width = 0.6) +
  geom_hline(data = mean_user_counts, aes(yintercept = mean_count, color = User), linetype = "dashed", size = 1) +
  scale_fill_manual(values =color_vector) +
  scale_color_manual(values = color_vector) +
  labs(title = "Email Count by User per Month",
       x = "Month",
       y = "Count",
       fill = "User") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 22), 
        axis.title = element_text(size = 24),
        axis.text.y = element_text(size = 22),
        plot.title = element_text(hjust = 0.5, size = 32),
#        legend.position = c(1.01, 0.2),       
        legend.position = 'none',
        legend.justification = c(0, 0),   
        legend.box = "vertical",
        plot.margin = margin(10, 100, 20, 10)) + 
  coord_cartesian(clip = "off")

color_vector = setNames(Usercolors$UserColor, Usercolors$User)
UserSummary |>
  group_by(User) |>
  summarise(Count = sum(Count)) |>
  ggplot(aes(x = reorder(User, -Count), y = Count, fill = User)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.7), width = 0.6) +
  scale_fill_manual(values =color_vector) +
  scale_color_manual(values = color_vector) +
  labs(title = "Email Count by User",
       x = "User",
       y = "Count",
       fill = "User") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 22), 
        axis.title = element_text(size = 24),
        axis.text.y = element_text(size = 22),
        plot.title = element_text(hjust = 0.5, size = 32),
        #        legend.position = c(1.01, 0.2),       
        legend.position = 'none',
        legend.justification = c(0, 0),   
        legend.box = "vertical",
        plot.margin = margin(10, 100, 20, 10)) + 
  coord_cartesian(clip = "off")

