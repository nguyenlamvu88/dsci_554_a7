
# DSCI 554 Assignment 7: Countries Dashboard with Color Scales, Maps, and Layouts
Vu Nguyen
USCID: 2120314402

An interactive data dashboard built with **React** and **D3.js** to visualize various datasets across multiple visual components. This project allows users to explore data through maps and charts, displaying information on demographics, health expenditure, GDP, population, migration, trade, and more. The datasets are meticulously sourced, cleaned, and tailored with manual and ChatGPT assistance for consistency and visualization readiness. All datasets are uploaded to GitHub and accessed via URLs, enabling seamless integration and dynamic, interactive visualizations.

## Table of Contents

- [Project Structure](#project-structure)
- [Data Overview](#data-overview)
  - [1. Population Data](#1-population-data)
  - [2. Health Expenditure Data](#2-health-expenditure-data)
  - [3. Current GDP Data](#3-current-gdp-data)
  - [4. Demographic Data](#4-demographic-data)
  - [5. Migration Data](#5-migration-data)
  - [6. Trade Data](#6-trade-data)
- [Dependencies and Configuration Files](#dependencies-and-configuration-files)
- [Usage](#usage)
- [Design Methodology for Maps, Charts, Graphs, and Visualizations](#design-methodology-for-maps-charts-graphs-and-visualizations)
- [Styling & Customization](#styling--customization)
- [Data Sources](#data-sources)
- [AI Assistance](#ai-assistance)

## Project Structure

The project follows a modular structure for maintainability and scalability:

```bash
A6-NGUYENLAMVU88/
├── .vscode/                     # Visual Studio Code configuration files
├── node_modules/                # Project dependencies
├── public/                      # Public folder for static assets
│   └── data/
│       ├── processed/           # Processed data files for visualization
│       │   ├── choroplethmap_health_expenditure.csv
│       │   ├── dot_map_populations_cities.csv
│       │   ├── gdp.csv
│       │   ├── hierarchical_demographic_zoomable.json
│       │   ├── import_export_cleaned.csv
│       │   └── merged_selected_countries_population_gdp.csv
│       └── raw/                 # Raw data files before processing
│           ├── API_NY.GDP.MKTP.CD_DS2_en_csv_v2_9865.csv
│           ├── API_SH.XPD.CHEX.GD.ZS_DS2_en_csv_v2_10468.csv
│           ├── SYB66_123_202310_Total Imports Exports and Balance of Trade.csv
│           ├── t1_TOP100.xlsx
│           ├── worldcities.csv
│           └── WPP2024_GEN_F01_DEMOGRAPHIC_INDICATORS_COMPACT.xlsx
├── src/
│   ├── components/
│   │   ├── layouts/             # Chart components for various layouts
│   │   │   ├── ChordDiagram.js
│   │   │   ├── DifferenceChart.js
│   │   │   ├── ForceDirectedGraph.js
│   │   │   ├── MigrationMap.js
│   │   │   ├── MultiLineChart.js
│   │   │   ├── ParallelCoordinatesChart.js
│   │   │   ├── PieChart.js
│   │   │   ├── StackedBarChart.js
│   │   │   ├── Sunburst.js
│   │   │   ├── Treemap.js
│   │   │   └── ZoomableCirclePacking.js
│   │   ├── maps/                # Map components
│   │   │   ├── ChoroplethMap.js
│   │   │   ├── DotMap.js
│   │   │   └── ProportionalSymbolMap.js
│   │   ├── Tooltip.css          # Tooltip styling for components
│   │   ├── Tooltip.js           # Tooltip component
│   │   ├── Dashboard.js         # Main dashboard component
│   │   └── Navbar.js            # Navigation bar component
│   ├── index.css                # Global styles
│   └── index.js                 # Entry point
├── .babelrc                     # Babel configuration
├── package-lock.json            # Lockfile for npm dependencies
├── package.json                 # Project configuration and dependencies
└── webpack.config.js            # Webpack configuration
```

## Data Overview

Each dataset was sourced, cleaned, and tailored with manual and ChatGPT assistance for consistency and visualization readiness, creating a structured view of global trends in population, health expenditure, GDP, demographics, migration, and trade. All datasets are uploaded to GitHub and accessed via URLs for seamless integration.

### 1. Population Data
- **Source**: [Simple Maps World Cities Database](https://simplemaps.com/data/world-cities)
- **Original File**: `worldcities.csv`
- **Processing**: Cleaned and processed with ChatGPT and manual adjustments, saved as `dot_map_populations_cities.csv` and uploaded to GitHub.
- **Final Use**:
  - **Dot Map**: Displays the top 10 populous cities in Brazil, China, Germany, India, and the United States.
  - **Treemap**: Compares population distributions among these cities, highlighting spatial differences.
  - **Pie Chart**: Visualizes the total population differences among the five countries.

### 2. Health Expenditure Data
- **Source**: [World Bank](https://data.worldbank.org/indicator/SH.XPD.CHEX.GD.ZS)
- **Original File**: `API_NY.GDP.MKTP.CD_DS2_en_csv_v2_9865.csv`
- **Processing**: Cleaned and processed with ChatGPT, saved as `choroplethmap_health_expenditure.csv`, and uploaded to GitHub.
- **Final Use**:
  - **Choropleth Map**: Shades countries based on health expenditure values, updating according to the selected year.
  - **Difference Chart**: Allows comparison of health expenditure between two countries.
  - **Stacked Bar Chart**: Displays health expenditure across countries or years, showing relative spending levels.

### 3. Current GDP Data
- **Source**: [World Bank](https://data.worldbank.org/indicator/NY.GDP.MKTP.CD)
- **Original File**: `API_SH.XPD.CHEX.GD.ZS_DS2_en_csv_v2_10468.csv`
- **Processing**: Cleaned and processed with ChatGPT and manual adjustments. Data for Brazil, China, Germany, India, and the United States from 2002 to 2023 was saved as `gdp.csv` and uploaded to GitHub. This file was merged with `dot_map_populations_cities.csv` to create `merged_selected_countries_population_gdp.csv`.
- **Final Use**:
  - **Proportional Symbol Map**: Combines population and GDP data for comparing population sizes and GDP values across the selected countries.

### 4. Demographic Data
- **Source**: [UN World Population Prospects](https://population.un.org/wpp/Download/Standard/CSV/)
- **Original File**: `WPP2024_GEN_F01_DEMOGRAPHIC_INDICATORS_COMPACT.csv`
- **Processing**: Cleaned and processed with ChatGPT and manual adjustments, saved as `forced_directed_graph_migration_data_with_top_countries.csv`, and transformed into `hierarchical_demographic_zoomable.json` for GitHub.
- **Final Use**:
  - **Zoomable Circle Packing & Sunburst Chart**: Allow users to explore nested demographic data, such as age, gender, and mortality breakdowns within populations.

### 5. Migration Data
- **Source**: [OECD Database on Immigrants](https://www.oecd.org/en/data/datasets/database-on-immigrants-in-oecd-and-non-oecd-countries.html)
- **Original File**: `t1_TOP100.csv`
- **Processing**: Cleaned and processed with ChatGPT, saved as `forced_directed_graph_migration_data_with_top_countries.csv` and transformed into `migration_data_parallel.json`. Both files were uploaded to GitHub.
- **Final Use**:
  - **Migration Map**: Visualizes migration routes and destinations on a map.
  - **Parallel Coordinates Chart**: Compares migration flows across multiple countries.
  - **Chord Diagram**: Shows bilateral migration flows between countries.
  - **Force Directed Graph**: Highlights top source and destination countries and migration linkages.

### 6. Trade Data
- **Source**: [UN Data – Total Imports, Exports, and Balance of Trade](https://data.un.org)
- **Original File**: `SYB66_123_202310_Total Imports Exports and Balance of Trade.csv`
- **Processing**: Cleaned and processed with ChatGPT and manual adjustments, saved as `import_export_cleaned.csv`, and uploaded to GitHub.
- **Final Use**:
  - **Multi-Line Chart**: Visualizes trade data for imports, exports, and trade balance over multiple years.

## Data Sources

All datasets are hosted on GitHub and accessed via raw URLs for seamless integration.

- **City Populations and Demographics**:
  - **URL**: [dot_map_populations_cities.csv](https://raw.githubusercontent.com/nguyenlamvu88/dsci_554_a7/main/dot_map_populations_cities.csv)
- **Health Expenditure**:
  - **URL**: [choroplethmap_health_expenditure.csv](https://raw.githubusercontent.com/nguyenlamvu88/dsci_554_a7/main/choroplethmap_health_expenditure.csv)
- **Population and GDP**:
  - **URL**: [merged_selected_countries_population_gdp.csv](https://raw.githubusercontent.com/nguyenlamvu88/dsci_554_a7/main/merged_selected_countries_population_gdp.csv)
- **Hierarchical Demographic Data**:
  - **URL**: [hierarchical_demographic_zoomable.json](https://raw.githubusercontent.com/nguyenlamvu88/dsci_554_a7/main/hierarchical_demographic_zoomable.json)
- **Trade Data**:
  - **URL**: [import_export_cleaned.csv](https://raw.githubusercontent.com/nguyenlamvu88/dsci_554_a7/main/import_export_cleaned.csv)
- **Migration Data**:
  - **URL**: [migration_data_parallel.json](https://raw.githubusercontent.com/nguyenlamvu88/dsci_554_a7/main/migration_data_parallel.json)

## Dependencies and Configuration Files

This project relies on several libraries and tools, and it includes configuration files to streamline development:

- **React**: For building user interface components.
- **D3.js**: For data manipulation and creating dynamic visualizations.
- **topojson-client**: For handling GeoJSON data.
- **d3-fetch**: For fetching CSV and JSON data.

### Configuration Files

Here’s how each of these files contributes to running React app on `localhost:3000`:

- **.babelrc**:
  - Configures Babel to compile modern JavaScript (ES6+ and JSX) into code that browsers can understand.
  - Specifies Babel presets, like `@babel/preset-env` and `@babel/preset-react`, to transform React and JavaScript features appropriately.

- **package.json** and **package-lock.json**:
  - `package.json`: Lists your project’s dependencies, scripts, and metadata. The `"start": "webpack serve --mode development --port 3000"` script tells Webpack to start a local server on port 3000 in development mode.
  - `package-lock.json`: Locks dependency versions to ensure consistency across installations.

- **index.html**:
  - Acts as the main HTML file and entry point for your application. HtmlWebpackPlugin uses this file as a template to inject your JavaScript bundle (created by Webpack) into the final HTML file served to the browser.

- **webpack.config.js**:
  - Configures Webpack to bundle your code. It includes loaders for handling files (like `babel-loader` for JSX) and plugins (like `HtmlWebpackPlugin`).
  - Specifies where to find `index.html` and tells Webpack how to bundle and serve files on the configured port (3000).

### Installation

To install the necessary dependencies, run:

```bash
npm install
```

## Usage

To start using the project, follow these steps:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/interactive-data-dashboard.git
   ```

2. **Navigate to the Project Directory**:

   ```bash
   cd interactive-data-dashboard
   ```

3. **Install Dependencies**:

   ```bash
   npm install
   ```

4. **Start the Development Server**:

   ```bash
   npm start
   ```

5. **View the Dashboard**:

   Open [http://localhost:3000](http://localhost:3000) in your browser to view the interactive dashboard.

### Interacting with the Dashboard

- **Navigation**: Use the sidebar to select different maps and charts.
- **Filters**: Utilize year and country selectors to dynamically adjust data displays.
- **Tooltips**: Hover over cities, regions, or chart elements to view detailed information.
- **Zoom & Pan**: Navigate maps with zoom and pan functionality for better exploration.

## Design Methodology for Maps, Charts, Graphs, and Visualizations

### Interactive Maps
- **Dot Map**: Displays population and demographic data of the top 10 populous cities in Brazil, China, Germany, India, and the United States.
- **Proportional Symbol Map**: Visualizes GDP, population, and trade data using varying symbol sizes and colors.
- **Choropleth Map**: Shades countries based on health expenditure values, updating according to the selected year.

### Diverse Charts and Graphs
- **Treemap**: Compares population distributions among top cities in selected countries, highlighting spatial differences.
- **Sunburst & Zoomable Circle Packing Charts**: Represent hierarchical demographic data by country, year, and gender.
- **Pie, Stacked Bar & Difference Charts**: Display comparative data on health expenditure, population distribution, and migration.
- **Force-Directed Graph**: Illustrates migration data with connections between top source and destination countries.
- **Chord Diagram**: Shows bilateral migration flows between countries.
- **Multi-Line Chart**: Visualizes trade data for imports, exports, and trade balance over multiple years.
- **Parallel Coordinates Chart**: Compares migration flows across multiple countries.

## Styling & Customization

- **Global Styles**: Located in `styles/index.css`, featuring a dark theme with complementary colors.
- **Component Styles**: Specific styles for components can be found in their respective CSS files (e.g., `Tooltip.css`).
- **Animations and Transitions**: Implemented for button hovers, zoom effects, and transitions to enhance user experience.
- **Tooltips**: Styled with a background color and shadow for readability; customize in `styles/Tooltip.css`.

## AI Assistance

The development of this project was supported by AI assistance, contributing to various components and functionalities. Below are summaries of the AI contributions:

- **AI Assistance 1**: Implemented a Dot Map in React using D3.js, displaying the top 10 most populous cities from selected countries. Addressed challenges related to data loading, circle sizing, color adjustments, and separation of closely positioned cities. Planned integration of Treemap and Pie Chart components for cohesive visualization.
  
- **AI Assistance 2**: Developed an interactive data dashboard featuring a DotMap and Treemap for visualizing city populations across different countries. Enhanced tooltips, legends, and styling for a cohesive look and improved user interaction.

- **AI Assistance 3**: Created a Zoomable Circle Packing chart using D3.js within a React component, displaying hierarchical data by country, gender, and population. Added year selection, unique color scales, and resolved runtime errors for enhanced visualization clarity.

- **AI Assistance 4**: Integrated a Zoomable Circle Packing visualization with a Proportional Symbol Map, refining data loading, tooltip functionality, and color settings for an interactive visual experience.

- **AI Assistance 5**: Integrated the ZoomableCirclePacking component into `Dashboard.js`, processed demographic data into hierarchical format, and ensured seamless interaction and state management across the dashboard.

- **AI Assistance 6**: Enhanced the Proportional Symbol Map with dynamic GDP representations, color-coded populations, year sliders, and improved tooltips and legends for better visual appeal and functionality.

- **AI Assistance 7**: Developed various data visualizations for immigration and trade datasets, including Multi-Line Charts, Parallel Coordinates Charts, Chord Diagrams, and Force Directed Graphs. Enhanced tooltip implementation, data filtering, and visual clarity.

- **AI Assistance 8**: Developed Migration Map and Force Directed Graph components, integrated migration flows with interactive features, addressed runtime errors, and refined code for improved functionality and visual representation.

- **AI Assistance 9**: Streamlined visualization selection by implementing a centralized `handleMapSelection` function, renamed buttons for clarity, and optimized component display logic for an intuitive user experience.

- **AI Assistance 10**: Developed a Migration Map component from scratch, integrating migration and GeoJSON datasets, normalized country names, implemented interactive tooltips, zoom/pan functionality, legends, and ensured responsive design within the Dashboard component.
