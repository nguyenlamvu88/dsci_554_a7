
# Interactive Data Dashboard

An interactive data dashboard built with React and D3.js to visualize various datasets across multiple visual components. This project allows users to explore data through maps and charts, displaying information on demographics, health expenditure, GDP, population, migration, and more.

## Table of Contents
- [Features](#features)
- [Design Methodology](#design-methodology)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Data Sources](#data-sources)
- [Usage](#usage)
- [Styling & Customization](#styling--customization)
- [Future Improvements](#future-improvements)

## Features

### Multiple Maps:
- **Dot Map**: Displays population and demographic data of top cities in each selected country.
- **Proportional Symbol Map**: Visualizes GDP, population, and trade data using varying symbol sizes and colors.
- **Choropleth Map**: Shows health expenditure per country, with intensity representing expenditure levels.

### Diverse Charts and Graphs:
- **Treemap**: Shows top populous cities by country.
- **Sunburst & Zoomable Circle Packing Charts**: Represent hierarchical demographic data by country, year, and gender.
- **Pie, Stacked Bar & Difference Charts**: Display comparative data on health expenditure, population distribution, and migration.
- **Force-Directed Graph**: Illustrates migration data with connections between top countries.

### Interactive Elements:
- Tooltips on hover for detailed information.
- Zoom & Pan functionality on maps.
- Year and country selectors to dynamically adjust data on maps and charts.

### Responsive & Modular Design:
- Adapts to screen sizes, with sidebar navigation and content panels.
- Dark theme with animations and transitions for a polished experience.
- 
## Design Methodology

The choice of maps, charts, and graphs in the Interactive Data Dashboard is grounded in ease of interpretability, effectiveness in communicating complex data relationships, and alignment with specific data types. Each visualization was selected to make the data exploration process intuitive and to enable users to gain insights quickly.

### Maps

1. **Dot Map**
   - **Purpose**: Visualize population concentrations within selected countries by plotting each city with a dot size proportional to its population.
   - **Data Suitability**: Location-based data like city population sizes, where each dot is both a precise location and a quantity.

2. **Proportional Symbol Map**
   - **Purpose**: Overlay circles on countries based on population, with color intensity reflecting GDP values. This enables dual comparisons of population size and economic output.
   - **Data Suitability**: Ideal for comparing data across regions of varying sizes, where larger symbols indicate greater magnitude.

3. **Choropleth Map**
   - **Purpose**: Display health expenditure across countries, with color intensity representing expenditure levels.
   - **Data Suitability**: Suitable for regional or national data with continuous values, such as percentages or rates.

### Charts & Graphs

4. **Treemap**
   - **Purpose**: Show city populations, grouped by country, highlighting how urban areas contribute to national population sizes.
   - **Data Suitability**: Effective for visualizing part-to-whole relationships within hierarchical data.

5. **Sunburst Chart**
   - **Purpose**: Display population data by country, year, and demographic segments in a hierarchical format.
   - **Data Suitability**: Best for hierarchical or multi-level data, enabling drill-down exploration.

6. **Zoomable Circle Packing**
   - **Purpose**: Show population data by gender within each country, with zoom functionality.
   - **Data Suitability**: Effective for nested or grouped data requiring visual hierarchy without complex comparisons.

7. **Pie Chart**
   - **Purpose**: Break down a country’s total population by its top cities to visualize relative size and distribution.
   - **Data Suitability**: Ideal for categorical data that represent parts of a whole, with limited categories for comparison clarity.

8. **Stacked Bar Chart**
   - **Purpose**: Visualize health expenditure across countries over time, comparing category contributions to a total.
   - **Data Suitability**: Suited for time series data with categories where both individual and cumulative values are important.

9. **Difference Chart**
   - **Purpose**: Highlight expenditure differences between two countries over time.
   - **Data Suitability**: Best for paired comparisons, showing trends and deviations.

10. **Force-Directed Graph**
    - **Purpose**: Visualize migration connections between countries, with line thickness representing migration volume.
    - **Data Suitability**: Suitable for network data, emphasizing connection strength between entities.

11. **Multi-Line Chart**
    - **Purpose**: Display trade data trends, such as imports and exports, over time.
    - **Data Suitability**: Time series data where trends and changes are essential.

12. **Parallel Coordinates Chart**
    - **Purpose**: Show migration data across metrics like origin, destination, age, and gender.
    - **Data Suitability**: Effective for multi-dimensional comparisons across multiple variables.

13. **Chord Diagram**
    - **Purpose**: Visualize migration flows between countries, with arc thickness proportional to migration volume.
    - **Data Suitability**: Ideal for data with flow or connection strength between entities.

14. **Migration Map**
    - **Purpose**: Show migration flows between countries with directional markers.
    - **Data Suitability**: Geographic data focused on movement or flow between locations.

- **City Populations and Demographics**: Used in Dot Map and Pie Chart for city-wise population visualizations.
  - Source: [City Data CSV](https://raw.githubusercontent.com/nguyenlamvu88/dsci_554_a7/main/dot_map_populations_cities.csv)

- **Health Expenditure**: Data for Choropleth Map and Stacked Bar Chart.
  - Source: [Health Expenditure CSV](https://raw.githubusercontent.com/nguyenlamvu88/dsci_554_a7/main/choroplethmap_health_expenditure.csv)

- **Population and GDP**: For Proportional Symbol Map displaying GDP and population.
  - Source: [Population and GDP CSV](https://raw.githubusercontent.com/nguyenlamvu88/dsci_554_a7/main/merged_selected_countries_population_gdp.csv)

- **Hierarchical Demographic Data**: For Sunburst and Zoomable Circle Packing visualizations.
  - Source: [Hierarchical Data JSON](https://raw.githubusercontent.com/nguyenlamvu88/dsci_554_a7/main/hierarchical_demographic_zoomable.json)

- **Trade Data**: For Multi-Line and Difference Charts.
  - Source: [Trade Data CSV](https://raw.githubusercontent.com/nguyenlamvu88/dsci_554_a7/main/import_export_cleaned.csv)

- **Migration Data**: For Migration Map, Parallel Coordinates, and Chord Diagram.
  - Source: [Migration Data JSON](https://raw.githubusercontent.com/nguyenlamvu88/dsci_554_a7/main/migration_data_parallel.json)

---

## Usage
To start using the project:

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/interactive-data-dashboard.git
   ```

2. Navigate to the project directory:

   ```bash
   cd interactive-data-dashboard
   ```

3. Install dependencies:

   ```bash
   npm install
   ```

4. Start the development server:

   ```bash
   npm start
   ```

5. Open `http://localhost:3000` in your browser to view the dashboard.

---

## Styling & Customization
- **Global Styles**: Found in `index.css`, featuring a dark theme with complementary colors.
- **Animations and Transitions**: Button hovers, zoom effects, and transitions are styled for a smoother experience.
- **Tooltip Styling**: Tooltips have a background color and shadow for readability. Customize tooltip styles in `Tooltip.css`.

---

## Future Improvements
- **Enhanced Interactivity**: Add filtering options for countries or data points.
- **Data Upload Feature**: Allow users to upload custom datasets for dynamic chart generation.
- **Export Options**: Enable data export (CSV/JSON) and chart snapshots.
- **Additional Charts**: Integrate more D3 visualizations, such as Heatmaps or Radar charts, for expanded analysis.

---

## Setup Instructions
Follow these steps to set up your project environment from scratch:

1. Clone the repository:

   ```bash
   git clone https://github.com/DSCI-554/a6-nguyenlamvu88.git
   ```

2. Navigate into the project directory:

   ```bash
   cd a6-nguyenlamvu88
   ```

3. Install the project dependencies:

   ```bash
   npm install
   ```

   This command installs all the necessary packages and dependencies listed in `package.json`.

---

### Installation of Required Loaders and Plugins
To ensure everything works correctly, install the necessary loaders and plugins:

```bash
npm install webpack webpack-cli webpack-dev-server babel-loader @babel/core @babel/preset-env @babel/preset-react html-webpack-plugin style-loader css-loader html-loader gh-pages --save-dev
```

---

### Deployment Steps
In the terminal, enter these commands:

1. **Running the Development Server**

   To start the development server and view the dashboard locally:

   ```bash
   npm start
   ```

   Open your browser and go to `http://localhost:3000` to view the dashboard. This command will start the Webpack dev server, allowing you to make changes to your code and see the updates in real-time.

2. **Build the project**:

   ```bash
   npm run build
   ```

3. **Deploy to GitHub Pages**:

   ```bash
   npm run deploy
   ```

   After running this command, your dashboard will be deployed to GitHub Pages. Once the deployment is complete, you can access your live dashboard at the following URL:

   `https://nguyenlamvu88.github.io/a6-nguyenlamvu88/`

---

### Creating Configuration Files
The following configuration files are created and posted in the repository:

1. **Create webpack.config.js**

   This file is the configuration for Webpack, defining how your project is bundled. It specifies the entry point, output directory, module rules for processing different file types, and the plugins used to generate the HTML file.

2. **Create .babelrc File**

   This file contains the configuration for Babel, which is used to transpile modern JavaScript and React code into a backward-compatible version. It specifies the presets that Babel should use for transforming the code.

3. **Update package.json**

   This file manages your project's dependencies and scripts. The scripts section should be updated to include commands for starting the development server, building the project for production, and deploying to GitHub Pages.

These files are essential for ensuring that the project builds and runs correctly and are included in the repository for reference.

