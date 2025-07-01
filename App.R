library(shiny)
library(shinydashboard)
library(dplyr)
library(lubridate)
library(plotly)
library(shinyWidgets)
library(DT)

# Load Data
readRenviron("credentials.renviron")
source("./Email Data.R")
source("./Phone Data.R")

emaildata <- emaildata %>%
  mutate(ID = as.character(EmailID))

# Data Cleaning
phonedata <- phonedata %>%
  mutate(Source = "Phone") %>%
  filter(!is.na(User) & !is.na(Source) & !is.na(Date)) %>%
  mutate(DateReceived = Date) %>%
  filter(User %in% Usercolors$User) %>%
  mutate(Y = '', Subject = '', Body = '', Subfolder = '', Attachments = '', ChainID = '', EmailID = '', Month = format(DateReceived, "%m"), 
         Year = format(DateReceived, "%Y"), ID = as.character(CallID)) %>%
  select(Y, DateReceived, Subject, Body, Subfolder, Attachments, ChainID, Source, EmailID, ID, Direction, User, Month, Year, Duration)

datacombined <- emaildata %>%
  mutate(Source = 'Email', CallID = '') %>%
  mutate(User = as.character(Category), Source = as.character(Source)) %>%  
  filter(!is.na(User) & !is.na(Source)) %>%
  mutate(DateReceived = as.Date(DateReceived), EmailID = as.character(EmailID), Duration = 0, Subject = str_trim(Subject), Body = str_trim(Body)) %>%
  filter(!is.na(DateReceived)) %>%
  filter(User %in% Usercolors$User) %>%
  select(Y, DateReceived, Subject, Body, Subfolder, Attachments, ChainID, Source, ID, Direction, User, Month, Year, Duration) %>%
  bind_rows(phonedata)

user_choices <- sort(unique(datacombined$User))

# UI
ui <- dashboardPage(
  skin = 'blue',
  dashboardHeader(
    title = tags$span(
      "OPERATIONS COMMUNICATION DASHBOARD",
      style = "display: block; width: 100%; text-align: center; font-weight: bold; font-size: 20pt;"
    ),
    titleWidth = "100%"
  ),
  dashboardSidebar(
    width = 250,
    dateRangeInput(
      inputId = "date_range",
      label = "Select Date Range:",
      start = as.Date("2025-01-01"),
      end = Sys.Date(),
      min = min(datacombined$DateReceived),
      max = Sys.Date(),
      format = "mm/dd/yyyy",
      separator = " to "
    ),
    sidebarMenu(
      menuItem("Filters", tabName = "filters", icon = icon("filter")),
      selectInput("selected_users", "Select Users:", 
                  choices = unique(user_choices), 
                  selected = NULL,   
                  multiple = TRUE),
      br(),
      uiOutput("custom_legend")
    )
  ),
  dashboardBody(
    tags$style(HTML('
      /* Sidebar Background Color */
      .main-sidebar, .main-sidebar .sidebar-menu, .main-sidebar .sidebar-menu li, .main-sidebar .sidebar-menu li a {
        background-color: #0F7DBC !important;
        color: white !important;
      }
      
      /* Remove Filters box background */
      .main-sidebar .sidebar-menu .active a,
      .main-sidebar .sidebar-menu .active {
        background-color: #0F7DBC !important;
        border-left-color: #0F7DBC !important;
      }

      /* Dashboard Header Background Color - FIXED */
      .skin-blue .main-header .navbar {
        background-color: #0F7DBC !important;
      }
      .skin-blue .main-header .logo {
        background-color: #0F7DBC !important;
      }

      .box-header {
        background-color: #009EED !important;
        color: white !important;
      }
        /* Increase dashboard header height */
      .skin-blue .main-header .navbar {
        min-height: 80px !important;
      }
    
      .skin-blue .main-header .navbar .navbar-custom-menu, 
      .skin-blue .main-header .navbar .navbar-right,
      .skin-blue .main-header .logo {
        height: 80px !important;
      }
    
      .skin-blue .main-sidebar {
        padding-top: 80px !important;
      }
      .box-title { color: white !important; }
        /* Application background color to white */
      .content-wrapper, .right-side {
        background-color: #FFFFFF !important;
      }
      .skin-blue .main-header .logo, 
      .skin-blue .main-header .navbar {
        height: 80px !important; /* adjust if needed */
      }
      
      .skin-blue .main-header .logo span {
        line-height: 80px !important; /* match header height */
      }
      /* Fix calendar popup being cut off in dateRangeInput */
      .daterangepicker {
          position: fixed !important;
          top: 100px !important; /* Adjust if needed */
          z-index: 9999 !important; 
      }
       /* Custom styling for the Count/Duration buttons */
      .btn-group-toggle .btn {
        background-color: #f2f2f2;
        color: #333;
        font-size: 14px;
        font-weight: bold;
        border: 1px solid #ccc;
        padding: 6px 14px;
        border-radius: 6px;
        transition: all 0.2s ease;
      }
    
      .btn-group-toggle .btn.active {
        background-color: #0F7DBC !important;
        color: #fff !important;
        border-color: #0F7DBC !important;
      }
    
      .btn-group-toggle .btn:hover {
        background-color: #d9eaff;
        color: #0F7DBC;
      }
    ')),
    fluidRow(
      column(
        width = 6,
        box(
          width = NULL,
          title = "Team Tasks",
          solidHeader = TRUE,
          status = "info",
          height = 150,
          div(
            style = "height: 100%; display: flex; justify-content: space-around; align-items: center; font-size: 18px;",
            div(style = "flex: 1; text-align: center; display: flex; flex-direction: column; gap: 6px; padding-top: 30px; justify-content: center; align-items: center; height: 100%;", 
                span(
                  tags$b("TOTAL"),
                  uiOutput("total_tasks")
                )
            ),
            div(style = "flex: 1; text-align: center; display: flex; flex-direction: column; gap: 6px; padding-top: 30px; justify-content: center; align-items: center; height: 100%;", 
                span(
                  tags$b("PHONE"),
                  HTML('<button id="phone_coachmark" style="margin-left: 0px; background: none; border: none; cursor: pointer;"><span style="font-size: 18px; display: inline-flex; align-items: center; justify-content: center; width: 14px; height: 14px; border-radius: 50%; border: 1px solid #007bff; color: #007bff;">i</span></button><div id="phone_tooltip" style="display: none; position: absolute; background: white; border: 1px solid #ccc; padding: 10px; border-radius: 8px; z-index: 1000;">Data Displayed with Format: <b>Incoming|Outgoing</b></div><script>document.getElementById("phone_coachmark").onclick = function(event) {var tooltip = document.getElementById("phone_tooltip");tooltip.style.display = tooltip.style.display === "none" ? "block" : "none";event.stopPropagation();};document.addEventListener("click", function(event) {var tooltip = document.getElementById("phone_tooltip");if (tooltip.style.display === "block" && !event.target.closest("#phone_coachmark")) {tooltip.style.display = "none";}});</script>')
                ),
                uiOutput("phone_tasks")
            ),
            div(style = "flex: 1; text-align: center; display: flex; flex-direction: column; gap: 6px; padding-top: 30px; justify-content: center; align-items: center; height: 100%;", 
                span(
                  tags$b("EMAIL"),
                  HTML('<button id="email_coachmark" style="margin-left: 0px; background: none; border: none; cursor: pointer;"><span style="font-size: 18px; display: inline-flex; align-items: center; justify-content: center; width: 14px; height: 14px; border-radius: 50%; border: 1px solid #007bff; color: #007bff;">i</span></button><div id="email_tooltip" style="display: none; position: absolute; background: white; border: 1px solid #ccc; padding: 10px; border-radius: 8px; z-index: 1000;">Data Displayed with Format: <b>Incoming|Outgoing</b></div><script>document.getElementById("email_coachmark").onclick = function(event) {var tooltip = document.getElementById("email_tooltip");tooltip.style.display = tooltip.style.display === "none" ? "block" : "none";event.stopPropagation();};document.addEventListener("click", function(event) {var tooltip = document.getElementById("email_tooltip");if (tooltip.style.display === "block" && !event.target.closest("#email_coachmark")) {tooltip.style.display = "none";}});</script>')
                ),
                uiOutput("email_tasks")
              )
          )
        ),
        box(
          width = NULL,
          title = "Performance Over Time",
          solidHeader = TRUE,
          status = "primary",
          height = 350,
          uiOutput("line_chart_ui")  # dynamically rendered
        )
      ),
      column(
        width = 6,
        box(
          width = NULL,
          height = 520,
          title = "Overall Performance",
          solidHeader = TRUE,
          status = "info",
          br(),
          plotlyOutput("donut_chart")
        )
      )
    ),
    fluidRow(
      column(
        width = 6,
        box(
          width = NULL,
          height = 550,
          title = "Emails by User",
          solidHeader = TRUE,
          status = "primary",
          div(
            style = "padding-top: 50px;", 
            plotlyOutput("email_bar_chart", height = "425px"))
        )
      ),
      column(
        width = 6,
        box(
          width = NULL,
          height = 550,
          title = "Phone Calls by User",
          solidHeader = TRUE,
          status = "primary",
          div(
            style = "display: flex; justify-content: flex-end; margin-bottom: 5px;",
            radioGroupButtons(
              inputId = "phone_metric",
              label = NULL,
              choices = c("Count", "Duration"),
              selected = "Count",
              direction = "horizontal",
              justified = FALSE,
              individual = TRUE,
              size = "xs"
            )
          ),
          plotlyOutput("phone_bar_chart")
        )
      )
    )
  )
)
# Server
server <- function(input, output, session) {
  clicked_user <- reactiveVal(NULL)
  clicked_direction <- reactiveVal(NULL)
  
  filtered_data <- reactive({
    req(input$date_range)
    
    df <- datacombined %>%
      filter(DateReceived >= as.Date(input$date_range[1]) & DateReceived <= as.Date(input$date_range[2]))
    if (!is.null(input$selected_users) && length(input$selected_users) > 0) {
      df <- df %>% filter(User %in% input$selected_users)
    }
    print(paste("Date Range:", input$date_range[1], "to", input$date_range[2]))
    
    df
  })
  output$line_chart_ui <- renderUI({
    plotlyOutput("time_line_chart", height = "280px")
  })
  
  output$custom_legend <- renderUI({
    legend_items <- Usercolors %>% arrange(User)
    tagList(
      tags$div(style = "padding: 10px 15px; font-weight: bold; color: white;", "User Key:"),
      lapply(seq_len(nrow(legend_items)), function(i) {
        tags$div(style = sprintf("display: flex; align-items: center; padding: 3px 15px; color: white;"),
                 tags$div(style = sprintf("width: 15px; height: 15px; background-color: %s; margin-right: 10px; border-radius: 3px;", legend_items$UserColor[i])),
                 tags$span(legend_items$User[i])
        )
      })
    )
  })
  
  
  output$total_tasks <- renderText({ nrow(filtered_data()) })
  
  output$phone_tasks <- renderUI({ 
    df <- filtered_data()
    incoming <- sum(df$Source == "Phone" & df$Direction == "Incoming", na.rm = TRUE)
    outgoing <- sum(df$Source == "Phone" & df$Direction == "Outgoing", na.rm = TRUE)
    
    HTML(paste0(
      '<span>', incoming, ' | ', outgoing, '</span>'
    ))
  })
  
  output$email_tasks <- renderUI({ 
    df <- filtered_data()
    incoming <- sum(df$Source == "Email" & df$Direction == "Incoming", na.rm = TRUE)
    outgoing <- sum(df$Source == "Email" & df$Direction == "Outgoing", na.rm = TRUE)
    
    HTML(paste0(
      '<span>', incoming, ' | ', outgoing, '</span>'
    ))
  })
  
  output$donut_chart <- renderPlotly({
    df <- filtered_data()
    
    if (nrow(df) == 0) return(NULL)
    
    df_summary <- df %>%
      group_by(User) %>%
      summarise(
        Count = n_distinct(ID),
        Email_Count = n_distinct(ID[Source == 'Email' & !is.na(ID)]),
        Phone_Count = n_distinct(ID[Source == 'Phone' & !is.na(ID)]),
        .groups = "drop"
      ) %>%
      mutate(
        Total = Count,
        Percentage = round((Count / sum(Count)) * 100, 2),
        TooltipText = paste0(
          "<b>", User, ": ", Percentage, "%</b><br>",
          "E-Mail: ", Email_Count, "<br>",
          "Phone Call: ", Phone_Count, "<br>",
          "Total: ", Total
        )
      )
    
    df_summary <- df_summary %>% arrange(User)
    df_summary$User <- factor(df_summary$User, levels = sort(unique(df_summary$User)))
    df_summary <- df_summary %>% inner_join(Usercolors, by = "User")
    
    plot_ly(
      df_summary, 
      labels = ~User, 
      values = ~Count, 
      type = 'pie', 
      textinfo = 'percent',
      text = ~TooltipText,
      hoverinfo = 'text',
      marker = list(colors = df_summary$UserColor),
      hole = 0.4  
    ) %>%
      layout(
        legend = list(
          categoryorder = "category ascending",
          font = list(size = 12),    
          yanchor = "top",
          y = 1
        ),
        showlegend = FALSE
      )
  })
  
  output$time_line_chart <- renderPlotly({
    df <- filtered_data()
    
    req(nrow(df) > 0)
    
    date_diff <- as.numeric(difftime(input$date_range[2], input$date_range[1], units = "days"))
    
    df <- df %>%
      mutate(DateGroup = if (date_diff <= 31) {
        as.Date(format(DateReceived, "%Y-%m-%d"))
      } else {
        floor_date(DateReceived, "month")
      })
    
    df_summary <- df %>%
      group_by(User, DateGroup) %>%
      summarise(Count = n_distinct(ID),
                Email_Count = n_distinct(ID[Source == 'Email']),
                Phone_Count = n_distinct(ID[Source == 'Phone']), 
                .groups = "drop") %>%
      inner_join(Usercolors, by = "User") %>%
      mutate(TooltipText = paste0(
        "User: ", User, "<br>",
        "Email: ", Email_Count, "<br>",
        "Phone: ", Phone_Count, "<br>",
        "Count: ", Count, "<br>"
      ))
    
    tick_vals <- sort(unique(df_summary$DateGroup))
    tick_text <- if (date_diff <= 31) {
      format(tick_vals, "%m-%d")
    } else {
      format(tick_vals, "%b %Y")
    }
    
    plot_ly(
      data = df_summary,
      x = ~DateGroup,
      y = ~Count,
      color = ~User,
      colors = df_summary$UserColor,
      type = 'scatter',
      mode = 'lines+markers',
      line = list(width = 4),
      hoverinfo = 'text',
      text = ~TooltipText
    ) %>%
      layout(
        xaxis = list(title = "Date", tickvals = tick_vals, ticktext = tick_text),
        yaxis = list(title = "Total Tasks"),
        showlegend = FALSE
      )
  })
  
  output$phone_bar_chart <- renderPlotly({
    df <- filtered_data() %>%
      filter(Source == "Phone")
    
    if (nrow(df) == 0) return(NULL)
    
    metric <- input$phone_metric
    x_axis_label <- if (metric == "Count") {
      "Count"
    } else {
      "Duration (Minutes)"
    }
    
    df_summary <- df %>%
      group_by(User) %>%
      summarise(
        Count = n_distinct(ID[!is.na(ID)]),
        Duration = sum(Duration, na.rm = TRUE) / 60,  # in minutes
        AvgDuration = ifelse(Count > 0, Duration / Count, 0),
        .groups = "drop"
      ) %>%
      arrange(desc(if (input$phone_metric == "Count") Count else Duration)) %>%
      inner_join(Usercolors, by = "User") %>%
      mutate(
        TooltipText = paste0(
          "User: ", User, "<br>",
          "Count: ", Count, "<br>",
          "Duration: ", round(Duration, 1), " min<br>",
          "Avg Call Length: ", round(AvgDuration, 1), " min"
        ),
        MetricValue = if (input$phone_metric == "Count") Count else Duration
      )
    
    df_summary$User <- factor(df_summary$User, levels = rev(df_summary$User))
    max_val <- max(df_summary$MetricValue, na.rm = TRUE)
    
    plot_ly(
      data = df_summary,
      x = ~MetricValue,
      y = ~User,
      type = 'bar',
      orientation = 'h',
      text = ~round(MetricValue, 1),
      hovertext = ~TooltipText,
      hoverinfo = "text",
      textposition = 'outside',
      cliponaxis = FALSE,
      marker = list(color = ~UserColor)
    ) %>%
      layout(
        margin = list(l = 0, r = 100),
        xaxis = list(
          title = x_axis_label,
          showgrid = FALSE,
          showline = FALSE,
          zeroline = TRUE,
          tickpadding = 20
        ),
        yaxis = list(
          title = "",
          showgrid = FALSE,
          showline = FALSE,
          zeroline = FALSE,
          ticks = '',
          showticklabels = FALSE
        ),
        showlegend = FALSE
      ) %>%
      add_annotations(
        x = -0.02 * max_val,
        y = df_summary$User,
        text = df_summary$User,
        xref = "x",
        yref = "y",
        showarrow = FALSE,
        xanchor = "right",
        font = list(size = 12)
      )
  })
  
  output$email_bar_chart <- renderPlotly({
    df <- filtered_data() %>%
      filter(Source == "Email")
    
    if (nrow(df) == 0) return(NULL)
    
    df_summary <- df %>%
      group_by(User) %>%
      summarise(TotalEmails = n_distinct(ID[Source == 'Email' & !is.na(ID)]), .groups = "drop") %>%
      arrange(desc(TotalEmails)) %>%
      inner_join(Usercolors, by = "User")
    
    df_summary$User <- factor(df_summary$User, levels = rev(df_summary$User))
    max_val <- max(df_summary$TotalEmails, na.rm = TRUE)
    
    base_plot <- plot_ly(
      data = df_summary,
      x = ~TotalEmails,
      y = ~User,
      type = 'bar',
      orientation = 'h',
      text = ~TotalEmails,
      textposition = 'outside',
      hoverinfo = "none",  # no hover on bars
      marker = list(color = ~UserColor),
      source = "emailChart"
    ) %>%
      layout(
        margin = list(l = 20, r = 60),
        xaxis = list(
          title = "",
          showgrid = FALSE,
          showline = FALSE,
          zeroline = TRUE,
          tickpadding = 20
        ),
        yaxis = list(
          title = "",
          showgrid = FALSE,
          showline = FALSE,
          zeroline = FALSE,
          ticks = '',
          showticklabels = FALSE  # suppress y-axis labels (we're adding our own)
        ),
        showlegend = FALSE
      )
    
    # Add name annotations
    for (i in seq_along(df_summary$User)) {
      base_plot <- base_plot %>% add_annotations(
        x = -0.02 * max(df_summary$TotalEmails),
        y = df_summary$User[i],
        text = df_summary$User[i],
        xref = "x",
        yref = "y",
        showarrow = FALSE,
        xanchor = "right",
        font = list(size = 12)
      )
    }
    base_plot <- base_plot %>%
      add_trace(
        data = df_summary,
        x = ~TotalEmails + max(TotalEmails) * 0.1,
        y = ~User,
        type = "scatter",
        mode = "markers",
        marker = list(
          symbol = "triangle-up",
          size = 16,
          color = "#40DB9E",
          line = list(width = 1, color = "#40DB9E")
        ),
        text = "Outgoing",
        hoverinfo = "text",
        textposition = "top center",  # helps push it upward a bit
        name = "Outgoing",
        source = "emailChart"
      ) %>%
      add_trace(
        data = df_summary,
        x = ~TotalEmails + max(TotalEmails) * 0.13,
        y = ~User,
        type = "scatter",
        mode = "markers",
        marker = list(
          symbol = "triangle-down",
          size = 16,
          color = "#D55B0D",
          line = list(width = 1, color = "#D55B0D")
        ),
        text = "Incoming",
        hoverinfo = "text",
        textposition = "bottom center",  # helps push it downward a bit
        name = "Incoming",
        source = "emailChart"
      )
    
    
    base_plot
  })
  
  observeEvent(event_data("plotly_click", source = "emailChart"), {
    click <- event_data("plotly_click", source = "emailChart")
    
    if (!is.null(click)) {
      clicked_user(click$y)
      
      direction_filter <- case_when(
        click$curveNumber == 1 ~ "Outgoing",
        click$curveNumber == 2 ~ "Incoming",
        TRUE ~ NA_character_
      )
      
      if (!is.na(direction_filter)) {
        clicked_direction(direction_filter)  # set direction clicked here
        
        showModal(modalDialog(
          title = paste(direction_filter, "Emails for", clicked_user()),
          tagList(
            tableOutput("user_table"),
            tags$div(
              style = "margin-top: 15px; text-align: left;",
              actionLink("show_more", "Show More")
            )
          ),
          easyClose = TRUE,
          footer = modalButton("Close")
        ))
        
        output$user_table <- renderTable({
          df <- filtered_data() %>%
            filter(
              Source == "Email",
              User == clicked_user(),
              Direction == direction_filter
            ) %>%
            count(Category = Y, name = "Count") %>%
            arrange(desc(Count))
          
          total_row <- data.frame(Category = "TOTAL", Count = sum(df$Count))
          bind_rows(df, total_row)
        })
      }
    }
  })
  
  
  observeEvent(input$show_more, {
    removeModal()
    showModal(modalDialog(
      title = paste("Detailed Emails for", clicked_user()),
      uiOutput("email_modal_content"),
      size = "l",
      easyClose = TRUE,
      footer = modalButton("Close")
    ))
    
    # Reactive value to track whether full Body is being shown
    show_full_body <- reactiveVal(FALSE)
    selected_row <- reactiveVal(NULL)
    
    # Render either the full table or full body based on user interaction
    output$email_modal_content <- renderUI({
      if (show_full_body()) {
        # Display full Body text
        fluidPage(
          actionButton("back_to_table", "Back to Table"),
          tags$hr(),
          verbatimTextOutput("full_body_text")
        )
      } else {
        # Show the data table
        DT::dataTableOutput("detailed_email_table")
      }
    })
    
    email_data <- reactive({
      req(clicked_direction(), clicked_user())
      
      filtered_data() %>%
        filter(
          Source == "Email",
          User == clicked_user(),
          Direction == clicked_direction()
        ) %>%
        mutate(ShortBody = ifelse(nchar(Body) > 200, paste0(substr(Body, 1, 200), "..."), Body)) %>%
        mutate(DateReceived = format(DateReceived, "%m/%d/%Y")) %>%
        transmute(
          Date = DateReceived,
          Subject,
          Body,  # Keep full Body for later access
          ShortBody,
          Attachments,
          Category = Y
        ) %>%
        arrange(desc(Date))
    })
    
    output$detailed_email_table <- DT::renderDataTable({
      datatable(
        email_data() %>% select(Date, Subject, ShortBody, Attachments, Category),
        selection = 'single',
        options = list(pageLength = 10, scrollX = TRUE),
        colnames = c("Date", "Subject", "Body", "Attachments", "Category")
      )
    })
    
    # Handle row click to show full Body
    observeEvent(input$detailed_email_table_rows_selected, {
      row <- input$detailed_email_table_rows_selected
      if (!is.null(row)) {
        selected_row(row)
        show_full_body(TRUE)
      }
    })
    
    # Render full Body text
    output$full_body_text <- renderText({
      if (!is.null(selected_row())) {
        email_data()[selected_row(), "Body", drop = TRUE]
      }
    })
    
    # Handle return to table
    observeEvent(input$back_to_table, {
      show_full_body(FALSE)
      selected_row(NULL)
    })
  })
}

shinyApp(ui, server)

