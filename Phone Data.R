# DocumentID = DocumentID
readRenviron("credentials.renviron")
# datafile <- setwd(Sys.getenv("datafile"))
phonedata <- readxl::read_excel('./Data/CR RawTeamsData.xlsx', sheet = 'CallQueueAnalyticsRaw')
phoneuser <- readxl::read_excel('./Data/CR RawTeamsData.xlsx', sheet = 'AgentTimelineAnalyticsRaw')
phoneduration <- readxl::read_excel('./Data/CR RawTeamsData.xlsx', sheet = 'fAgentTimelineAnalytics')
phonedata <- phonedata |>
  left_join(phoneuser, by = c('DocumentID' = 'DocumentID')) |>
  left_join(phoneduration, by = c('DocumentID' = 'DocumentID')) |>
  filter(!is.na(`Call Duration (Minutes)`))
#  mutate(CallQueueCallResult = ifelse(CallQueueCallResult == 'agent_joined_conference', 'Connected', ifelse(CallQueueCallResult == 'timed_out', 'Timed Out', 'Disconnected'))) |>
#  mutate(CallQueueCallResult = as.factor(CallQueueCallResult))
phonedata <- phonedata |>
  mutate(Date =as.POSIXct(str_extract(UserStartTimeUTC.x, "^[^T]+"), format = "%Y-%m-%d")) |>
  select(CallID = DocumentID, User = `Agent Name`, Date, Duration = CallQueueDurationSeconds) |>
  mutate(Direction = "Incoming", Duration = as.numeric(Duration))
# User, Date, Duration, Direction, ID, 
csv_files <- list.files(path = "Data", pattern = "Phone Data.*\\.csv$", full.names = TRUE)
phonedata_ind <- csv_files |>
  lapply(read.csv, stringsAsFactors = FALSE) |>
  bind_rows() |>
  mutate(Date = as.POSIXct(str_extract(Start.Time, "^[^T]+"), format = "%Y-%m-%d"), Duration.Seconds = as.numeric(Duration.Seconds)) |>
  select(CallID = Call.ID, User = User.Display.Name, Date, Duration = Duration.Seconds, Direction = Call.Direction)
phonedata <- phonedata |>
  bind_rows(phonedata_ind) |>
  mutate(Direction = ifelse(Direction == "Outbound", "Outgoing", "Incoming"), User = str_extract(User, pattern = "^[^ ]+")) %>%
  mutate(User = ifelse(User == 'arosario', 'Ariel', ifelse(User == 'crobinson', 'Christian', ifelse(User == 'damezquita', 'Denisse', 
                ifelse(User == 'ganderson', 'Gary', ifelse(User == 'gbrandon', 'Greg', ifelse(User == 'kporter', 'Kristian', 
                ifelse(User == 'mzhou', 'Ming', ifelse(User == 'nonks', 'Nathan', ifelse(User == 'pjeffries', 'Porschea', 
                ifelse(User == 'tprideaux', 'Tyler', User)))))))))))

