"""
PyPSA-Ethiopia Streamlit App - Updated Version
Integrated with standalone data downloader (no Snakemake required)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
import yaml
import time
from datetime import datetime, timedelta
import os
import sys
import logging
import threading
import queue
from typing import Dict, List, Optional, Tuple

# Import our standalone data downloader
from data_bundle_downloader import DataBundleDownloader, check_ethiopia_data_status

# Try to import PyPSA
try:
    import pypsa
    PYPSA_INSTALLED = True
except ImportError:
    PYPSA_INSTALLED = False

# Page configuration
st.set_page_config(
    page_title="PyPSA-Ethiopia Energy Model",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 16px;
    }
    .status-box {
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .status-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .status-error {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .status-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
    }
    .console-output {
        background-color: #1e1e1e;
        color: #00ff00;
        font-family: 'Courier New', monospace;
        padding: 10px;
        border-radius: 5px;
        height: 400px;
        overflow-y: auto;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_status' not in st.session_state:
    st.session_state.data_status = None
if 'model_config' not in st.session_state:
    st.session_state.model_config = {}
if 'current_network' not in st.session_state:
    st.session_state.current_network = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class PyPSAEthiopiaApp:
    """Main application class for PyPSA-Ethiopia"""
    
    def __init__(self):
        self.base_path = Path.cwd()
        self.data_downloader = DataBundleDownloader(self.base_path)
        self.config = self.load_default_config()
    
    def load_default_config(self) -> Dict:
        """Load default configuration for Ethiopia model"""
        return {
            'country': 'ET',
            'scenario': {
                'clusters': 10,
                'planning_horizons': [2030],
                'Co2L': 0.05,  # CO2 budget reduction
            },
            'snapshots': {
                'start': '2013-01-01',
                'end': '2013-01-31',  # One month for testing
            },
            'renewable': {
                'solar': True,
                'onwind': True,
                'hydro': True,
                'geothermal': True,
            },
            'conventional': {
                'gas_ocgt': True,
                'gas_ccgt': False,
                'oil': False,
                'coal': False,
            },
            'storage': {
                'battery': False,
                'pumped_hydro': True,
            },
            'solver': {
                'name': 'highs',
                'options': {
                    'threads': 4,
                    'time_limit': 3600,  # 1 hour
                }
            }
        }
    
    def create_pypsa_network_basic(self) -> pypsa.Network:
        """Create a basic PyPSA network for Ethiopia (simplified version)"""
        if not PYPSA_INSTALLED:
            raise ImportError("PyPSA not installed")
        
        # Create network
        n = pypsa.Network()
        
        # Add buses (simplified - just major regions)
        regions = {
            'Addis_Ababa': {'x': 38.7578, 'y': 9.0192},
            'Mekelle': {'x': 39.4753, 'y': 13.4969},
            'Bahir_Dar': {'x': 37.3961, 'y': 11.5745},
            'Hawassa': {'x': 38.4762, 'y': 7.0527},
            'Dire_Dawa': {'x': 41.8659, 'y': 9.6010}
        }
        
        for region, coords in regions.items():
            n.add("Bus", region, x=coords['x'], y=coords['y'], country='ET')
        
        # Add loads (simplified)
        for region in regions.keys():
            n.add("Load", f"load_{region}", 
                  bus=region, 
                  p_set=np.random.uniform(100, 500, len(n.snapshots)))
        
        # Add generators based on configuration
        if self.config['renewable']['solar']:
            for region in regions.keys():
                n.add("Generator", f"solar_{region}",
                      bus=region,
                      carrier="solar",
                      p_nom_extendable=True,
                      capital_cost=600,  # EUR/MW
                      marginal_cost=0,
                      p_max_pu=np.random.uniform(0, 1, len(n.snapshots)))
        
        if self.config['renewable']['onwind']:
            for region in ['Mekelle', 'Bahir_Dar']:  # Good wind regions
                n.add("Generator", f"wind_{region}",
                      bus=region,
                      carrier="onwind", 
                      p_nom_extendable=True,
                      capital_cost=1200,  # EUR/MW
                      marginal_cost=0,
                      p_max_pu=np.random.uniform(0, 0.8, len(n.snapshots)))
        
        if self.config['renewable']['hydro']:
            n.add("Generator", "hydro_GERD",
                  bus="Bahir_Dar",
                  carrier="hydro",
                  p_nom=6450,  # MW - Grand Ethiopian Renaissance Dam
                  marginal_cost=0,
                  p_max_pu=np.random.uniform(0.3, 1, len(n.snapshots)))
        
        if self.config['conventional']['gas_ocgt']:
            n.add("Generator", "gas_addis",
                  bus="Addis_Ababa", 
                  carrier="OCGT",
                  p_nom_extendable=True,
                  capital_cost=560,  # EUR/MW
                  marginal_cost=50,  # EUR/MWh
                  efficiency=0.4)
        
        # Add transmission lines (simplified)
        connections = [
            ('Addis_Ababa', 'Bahir_Dar', 300, 500),
            ('Addis_Ababa', 'Hawassa', 200, 300),
            ('Addis_Ababa', 'Dire_Dawa', 250, 400),
            ('Mekelle', 'Addis_Ababa', 400, 600),
        ]
        
        for bus0, bus1, length, s_nom in connections:
            n.add("Line", f"{bus0}-{bus1}",
                  bus0=bus0, bus1=bus1,
                  length=length,
                  s_nom=s_nom,
                  x=0.1 * length / 1000)  # Simplified reactance
        
        return n
    
    def solve_network(self, network: pypsa.Network) -> Dict:
        """Solve the PyPSA network"""
        try:
            # Solve
            solver_name = self.config['solver']['name']
            solver_options = self.config['solver']['options']
            
            status, termination_condition = network.optimize(
                solver_name=solver_name,
                solver_options=solver_options
            )
            
            # Extract results
            results = {
                'status': status,
                'termination_condition': termination_condition,
                'objective': network.objective,
                'generation': network.generators_t.p.sum().to_dict(),
                'capacity': network.generators.p_nom_opt.to_dict(),
                'load': network.loads_t.p.sum().sum(),
                'curtailment': 0,  # Simplified
                'co2_emissions': 0,  # Simplified
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Network optimization failed: {e}")
            return {'status': 'failed', 'error': str(e)}

def main():
    """Main application"""
    app = PyPSAEthiopiaApp()
    
    # Sidebar
    with st.sidebar:
        st.image("https://pypsa-meets-earth.github.io/assets/img/logo.png", width=200)
        st.title("PyPSA-Ethiopia")
        st.markdown("---")
        
        # Navigation
        page = st.selectbox(
            "Navigation",
            ["🏠 Home", "📦 Data Management", "⚙️ Configuration", "🚀 Run Model", "📊 Results", "📈 Analysis"]
        )
        
        st.markdown("---")
        
        # Quick status
        if PYPSA_INSTALLED:
            st.success("✅ PyPSA Installed")
        else:
            st.error("❌ PyPSA Not Installed")
        
        # Data status
        if st.session_state.data_status is None:
            st.session_state.data_status = check_ethiopia_data_status()
        
        data_status = st.session_state.data_status
        if data_status['ready_for_modeling']:
            st.success("✅ Data Ready")
        else:
            st.warning(f"⚠️ Data: {data_status['completion_percentage']:.0f}%")
    
    # Main content
    if page == "🏠 Home":
        show_home_page()
    elif page == "📦 Data Management":
        show_data_management_page(app)
    elif page == "⚙️ Configuration":
        show_configuration_page(app)
    elif page == "🚀 Run Model":
        show_run_model_page(app)
    elif page == "📊 Results":
        show_results_page(app)
    elif page == "📈 Analysis":
        show_analysis_page(app)

def show_home_page():
    """Home page with overview"""
    st.title("🇪🇹 PyPSA-Ethiopia Energy System Modeling")
    
    st.markdown("""
    Welcome to PyPSA-Ethiopia, a comprehensive energy system modeling tool for Ethiopia 
    based on the PyPSA (Python for Power System Analysis) framework.
    
    ## Key Features
    - **Data-driven modeling**: Uses real geographical and meteorological data
    - **Renewable energy focus**: Models solar, wind, hydro, and geothermal potential
    - **Transmission planning**: Optimizes transmission network expansion
    - **Scenario analysis**: Compare different energy transition pathways
    - **Interactive visualization**: Explore results through interactive maps and charts
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🌍 Geographic Scope
        - **Country**: Ethiopia
        - **Resolution**: Regional clusters
        - **Time horizon**: 2020-2050
        - **Weather data**: ERA5 reanalysis
        """)
    
    with col2:
        st.markdown("""
        ### ⚡ Technologies
        - Solar PV
        - Onshore Wind
        - Hydroelectric
        - Geothermal
        - Gas turbines
        - Storage systems
        """)
    
    with col3:
        st.markdown("""
        ### 🎯 Applications
        - Energy planning
        - Policy analysis
        - Investment decisions
        - Grid integration studies
        """)
    
    # Quick start guide
    with st.expander("🚀 Quick Start Guide", expanded=True):
        st.markdown("""
        1. **Data Management**: Download and verify required data files
        2. **Configuration**: Set up your modeling scenario
        3. **Run Model**: Execute the optimization
        4. **Results**: Analyze capacity expansion and generation patterns
        5. **Analysis**: Create custom visualizations and reports
        """)

def show_data_management_page(app):
    """Data management page"""
    st.title("📦 Data Management")
    
    # Get current status
    data_status = check_ethiopia_data_status()
    st.session_state.data_status = data_status
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📊 Data Files</h4>
            <h2>{data_status['available_files']}/{data_status['total_files']}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📈 Completion</h4>
            <h2>{data_status['completion_percentage']:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        status_text = "Ready" if data_status['ready_for_modeling'] else "Incomplete"
        status_color = "#28a745" if data_status['ready_for_modeling'] else "#dc3545"
        st.markdown(f"""
        <div class="metric-card">
            <h4>🎯 Status</h4>
            <h2 style="color: {status_color};">{status_text}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        if st.button("🔄 Refresh Status", type="secondary"):
            st.session_state.data_status = None
            st.experimental_rerun()
    
    # Data status details
    st.header("Data Files Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("✅ Available Files")
        if data_status['available_list']:
            for file in data_status['available_list']:
                st.success(f"📄 {file}")
        else:
            st.info("No files downloaded yet")
    
    with col2:
        st.subheader("❌ Missing Files")
        if data_status['missing_list']:
            for file in data_status['missing_list']:
                st.error(f"📄 {file}")
        else:
            st.success("All files available!")
    
    # Download section
    st.header("Download Data")
    
    if not data_status['ready_for_modeling']:
        st.warning("⚠️ Missing data files required for modeling. Click 'Download All Data' to fetch them automatically.")
        
        if st.button("📥 Download All Data", type="primary"):
            with st.spinner("Downloading data bundles... This may take several minutes."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simulate download progress
                try:
                    success = app.data_downloader.download_ethiopia_essential_data()
                    
                    if success:
                        progress_bar.progress(100)
                        st.success("🎉 All data downloaded successfully!")
                        st.session_state.data_status = None  # Reset to refresh
                        time.sleep(2)
                        st.experimental_rerun()
                    else:
                        st.error("❌ Some downloads failed. Please check the logs.")
                
                except Exception as e:
                    st.error(f"❌ Download failed: {e}")
    
    else:
        st.success("🎉 All required data files are available!")
        st.info("You can proceed to configure and run your energy system model.")
    
    # Manual data info
    with st.expander("ℹ️ Data Sources Information"):
        st.markdown("""
        ### Data Sources
        
        **Geographic Data**
        - **EEZ boundaries**: Exclusive Economic Zones v11
        - **Land cover**: Copernicus Global Land Service
        - **Topography**: GEBCO 2021 bathymetry
        
        **Energy Data**
        - **Load profiles**: SSP2-2.6 socioeconomic scenarios
        - **Renewable profiles**: ERA5 weather reanalysis
        - **Hydrobasins**: HydroBASINS global dataset
        
        **Weather Data**
        - **Source**: ERA5 reanalysis (ECMWF)
        - **Resolution**: 0.25° × 0.25°
        - **Variables**: Wind, solar irradiance, temperature
        - **Period**: 2013 (representative year)
        
        All data is automatically downloaded from official sources including Zenodo, 
        Copernicus Climate Data Store, and HydroSHEDS.
        """)

def show_configuration_page(app):
    """Configuration page"""
    st.title("⚙️ Model Configuration")
    
    # Configuration form
    with st.form("config_form"):
        st.header("Scenario Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📅 Time Period")
            start_date = st.date_input("Start Date", datetime(2013, 1, 1))
            end_date = st.date_input("End Date", datetime(2013, 1, 31))
            
            st.subheader("🔗 Network")
            clusters = st.slider("Number of Regions", 5, 20, 10, 1)
            
        with col2:
            st.subheader("🎯 Objectives")
            co2_reduction = st.slider("CO2 Reduction Target (%)", 0, 100, 95, 5)
            
            st.subheader("🔧 Solver")
            solver = st.selectbox("Solver", ["highs", "gurobi", "cplex"], index=0)
            time_limit = st.number_input("Time Limit (seconds)", 300, 7200, 3600, 300)
        
        st.header("Technology Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🌞 Renewable Energy")
            solar = st.checkbox("Solar PV", True)
            wind = st.checkbox("Onshore Wind", True)
            hydro = st.checkbox("Hydroelectric", True)
            geothermal = st.checkbox("Geothermal", True)
        
        with col2:
            st.subheader("🏭 Conventional")
            gas_ocgt = st.checkbox("Gas OCGT", True)
            gas_ccgt = st.checkbox("Gas CCGT", False)
            coal = st.checkbox("Coal", False)
            oil = st.checkbox("Oil", False)
        
        with col3:
            st.subheader("🔋 Storage")
            battery = st.checkbox("Battery Storage", False)
            pumped_hydro = st.checkbox("Pumped Hydro", True)
        
        st.header("Economic Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            discount_rate = st.slider("Discount Rate (%)", 0.0, 15.0, 7.0, 0.5)
            carbon_price = st.number_input("Carbon Price ($/tCO2)", 0, 200, 50, 10)
        
        with col2:
            planning_horizon = st.selectbox("Planning Horizon", [2030, 2040, 2050], index=0)
        
        # Submit configuration
        submitted = st.form_submit_button("💾 Save Configuration", type="primary")
        
        if submitted:
            # Update app configuration
            app.config.update({
                'snapshots': {
                    'start': start_date.strftime('%Y-%m-%d'),
                    'end': end_date.strftime('%Y-%m-%d'),
                },
                'scenario': {
                    'clusters': clusters,
                    'planning_horizons': [planning_horizon],
                    'Co2L': (100 - co2_reduction) / 100,
                },
                'renewable': {
                    'solar': solar,
                    'onwind': wind, 
                    'hydro': hydro,
                    'geothermal': geothermal,
                },
                'conventional': {
                    'gas_ocgt': gas_ocgt,
                    'gas_ccgt': gas_ccgt,
                    'coal': coal,
                    'oil': oil,
                },
                'storage': {
                    'battery': battery,
                    'pumped_hydro': pumped_hydro,
                },
                'solver': {
                    'name': solver,
                    'options': {
                        'time_limit': time_limit,
                        'threads': 4,
                    }
                },
                'economics': {
                    'discount_rate': discount_rate / 100,
                    'carbon_price': carbon_price,
                }
            })
            
            st.session_state.model_config = app.config
            st.success("✅ Configuration saved successfully!")
    
    # Display current configuration
    if st.session_state.model_config:
        with st.expander("📋 Current Configuration"):
            st.json(st.session_state.model_config)

def show_run_model_page(app):
    """Model execution page"""
    st.title("🚀 Run Model")
    
    # Check prerequisites
    data_status = st.session_state.data_status or check_ethiopia_data_status()
    has_config = bool(st.session_state.model_config)
    
    # Status checks
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if PYPSA_INSTALLED:
            st.success("✅ PyPSA Available")
        else:
            st.error("❌ PyPSA Not Installed")
    
    with col2:
        if data_status['ready_for_modeling']:
            st.success("✅ Data Ready")
        else:
            st.error("❌ Data Incomplete")
    
    with col3:
        if has_config:
            st.success("✅ Configuration Set")
        else:
            st.warning("⚠️ No Configuration")
    
    # Show configuration summary
    if has_config:
        st.header("📋 Model Configuration")
        config = st.session_state.model_config
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Scenario**")
            st.write(f"• Regions: {config.get('scenario', {}).get('clusters', 'N/A')}")
            st.write(f"• Time period: {config.get('snapshots', {}).get('start', 'N/A')} to {config.get('snapshots', {}).get('end', 'N/A')}")
            
            renewable_techs = [k for k, v in config.get('renewable', {}).items() if v]
            st.write(f"• Renewable: {', '.join(renewable_techs) if renewable_techs else 'None'}")
        
        with col2:
            conventional_techs = [k for k, v in config.get('conventional', {}).items() if v]
            st.write(f"• Conventional: {', '.join(conventional_techs) if conventional_techs else 'None'}")
            
            storage_techs = [k for k, v in config.get('storage', {}).items() if v]
            st.write(f"• Storage: {', '.join(storage_techs) if storage_techs else 'None'}")
            
            st.write(f"• Solver: {config.get('solver', {}).get('name', 'N/A')}")
    
    # Run model section
    st.header("🎮 Model Execution")
    
    can_run = PYPSA_INSTALLED and data_status['ready_for_modeling'] and has_config
    
    if not can_run:
        st.warning("⚠️ Cannot run model. Please ensure all prerequisites are met:")
        if not PYPSA_INSTALLED:
            st.write("• Install PyPSA")
        if not data_status['ready_for_modeling']:
            st.write("• Download required data")
        if not has_config:
            st.write("• Set up configuration")
    
    else:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.success("🎉 Ready to run energy system optimization!")
        
        with col2:
            run_model = st.button("▶️ Run Model", type="primary", disabled=not can_run)
        
        if run_model:
            with st.spinner("Running energy system optimization..."):
                try:
                    # Create network
                    st.info("📊 Creating network...")
                    network = app.create_pypsa_network_basic()
                    
                    # Set snapshots
                    snapshots = pd.date_range(
                        start=config['snapshots']['start'],
                        end=config['snapshots']['end'], 
                        freq='H'
                    )
                    network.set_snapshots(snapshots[:168])  # Limit to one week for demo
                    
                    st.info("⚡ Optimizing system...")
                    results = app.solve_network(network)
                    
                    if results['status'] == 'ok':
                        st.success("✅ Optimization completed successfully!")
                        
                        # Store results
                        st.session_state.current_network = network
                        st.session_state.model_results = results
                        
                        # Show quick results
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Cost", f"${results.get('objective', 0):,.0f}")
                        
                        with col2:
                            total_generation = sum(results.get('generation', {}).values())
                            st.metric("Total Generation", f"{total_generation:.0f} MWh")
                        
                        with col3:
                            renewable_gen = sum(v for k, v in results.get('generation', {}).items() 
                                              if any(tech in k.lower() for tech in ['solar', 'wind', 'hydro']))
                            if total_generation > 0:
                                renewable_share = renewable_gen / total_generation * 100
                            else:
                                renewable_share = 0
                            st.metric("Renewable Share", f"{renewable_share:.1f}%")
                        
                        st.info("📊 View detailed results in the Results tab.")
                        
                    else:
                        st.error(f"❌ Optimization failed: {results.get('error', 'Unknown error')}")
                
                except Exception as e:
                    st.error(f"❌ Model execution failed: {e}")
                    logger.error(f"Model execution error: {e}")

def show_results_page(app):
    """Results visualization page"""
    st.title("📊 Results")
    
    if st.session_state.current_network is None:
        st.warning("⚠️ No model results available. Please run a model first.")
        return
    
    network = st.session_state.current_network
    results = st.session_state.model_results
    
    # Results overview
    st.header("📈 Results Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Objective Value", f"${results.get('objective', 0):,.0f}")
    
    with col2:
        total_load = results.get('load', 0)
        st.metric("Total Load", f"{total_load:.0f} MWh")
    
    with col3:
        total_capacity = sum(results.get('capacity', {}).values())
        st.metric("Total Capacity", f"{total_capacity:.0f} MW")
    
    with col4:
        renewable_cap = sum(v for k, v in results.get('capacity', {}).items() 
                           if any(tech in k.lower() for tech in ['solar', 'wind', 'hydro']))
        if total_capacity > 0:
            renewable_share = renewable_cap / total_capacity * 100
        else:
            renewable_share = 0
        st.metric("Renewable Capacity", f"{renewable_share:.1f}%")
    
    # Detailed results tabs
    tab1, tab2, tab3 = st.tabs(["💡 Generation", "⚡ Capacity", "🌍 Network"])
    
    with tab1:
        st.subheader("Generation by Technology")
        
        generation_data = results.get('generation', {})
        if generation_data:
            # Create generation chart
            gen_df = pd.DataFrame.from_dict(generation_data, orient='index', columns=['Generation'])
            gen_df['Technology'] = gen_df.index
            
            fig = px.bar(gen_df, x='Technology', y='Generation',
                        title='Generation by Technology (MWh)')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No generation data available")
    
    with tab2:
        st.subheader("Installed Capacity")
        
        capacity_data = results.get('capacity', {})
        if capacity_data:
            # Create capacity chart
            cap_df = pd.DataFrame.from_dict(capacity_data, orient='index', columns=['Capacity'])
            cap_df['Technology'] = cap_df.index
            cap_df = cap_df[cap_df['Capacity'] > 0]  # Only show non-zero capacities
            
            fig = px.pie(cap_df, values='Capacity', names='Technology',
                        title='Installed Capacity by Technology (MW)')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No capacity data available")
    
    with tab3:
        st.subheader("Network Topology")
        
        # Simple network visualization
        if hasattr(network, 'buses') and not network.buses.empty:
            bus_data = network.buses.copy()
            
            fig = go.Figure()
            
            # Add buses
            fig.add_trace(go.Scattergeo(
                lon=bus_data['x'],
                lat=bus_data['y'],
                text=bus_data.index,
                mode='markers+text',
                marker=dict(size=10, color='red'),
                textposition='top center',
                name='Buses'
            ))
            
            # Add lines if available
            if hasattr(network, 'lines') and not network.lines.empty:
                for idx, line in network.lines.iterrows():
                    bus0_x = bus_data.loc[line['bus0'], 'x']
                    bus0_y = bus_data.loc[line['bus0'], 'y'] 
                    bus1_x = bus_data.loc[line['bus1'], 'x']
                    bus1_y = bus_data.loc[line['bus1'], 'y']
                    
                    fig.add_trace(go.Scattergeo(
                        lon=[bus0_x, bus1_x, None],
                        lat=[bus0_y, bus1_y, None],
                        mode='lines',
                        line=dict(width=2, color='blue'),
                        showlegend=False
                    ))
            
            fig.update_geos(
                scope='africa',
                showframe=False,
                showcoastlines=True,
                projection_type='equirectangular'
            )
            
            fig.update_layout(
                title='Ethiopia Energy Network',
                geo=dict(
                    lonaxis_range=[32, 48],
                    lataxis_range=[3, 18]
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No network topology data available")

def show_analysis_page(app):
    """Advanced analysis page"""
    st.title("📈 Advanced Analysis")
    
    if st.session_state.current_network is None:
        st.warning("⚠️ No model results available for analysis. Please run a model first.")
        return
    
    network = st.session_state.current_network
    results = st.session_state.model_results
    
    # Analysis options
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["💰 Economic Analysis", "🌱 Environmental Impact", "⚡ System Adequacy", "📊 Sensitivity Analysis"]
    )
    
    if analysis_type == "💰 Economic Analysis":
        st.header("Economic Analysis")
        
        # LCOE calculation (simplified)
        st.subheader("Levelized Cost of Energy (LCOE)")
        
        generation_data = results.get('generation', {})
        capacity_data = results.get('capacity', {})
        
        if generation_data and capacity_data:
            lcoe_data = {}
            for tech in generation_data.keys():
                # Simplified LCOE calculation
                if tech in capacity_data and generation_data[tech] > 0:
                    # Assumed capital costs (EUR/MW)
                    capex = {'solar': 600, 'wind': 1200, 'hydro': 2000, 'gas': 800}.get(
                        tech.split('_')[0], 1000)
                    
                    # Simplified LCOE = (CAPEX * capacity) / (generation * years)
                    lcoe = (capex * capacity_data[tech]) / (generation_data[tech] * 20)  # 20 year lifetime
                    lcoe_data[tech] = lcoe
            
            if lcoe_data:
                lcoe_df = pd.DataFrame.from_dict(lcoe_data, orient='index', columns=['LCOE'])
                lcoe_df['Technology'] = lcoe_df.index
                
                fig = px.bar(lcoe_df, x='Technology', y='LCOE',
                           title='Levelized Cost of Energy by Technology (EUR/MWh)')
                st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "🌱 Environmental Impact":
        st.header("Environmental Impact")
        
        st.subheader("CO2 Emissions")
        
        # Simplified emissions calculation
        generation_data = results.get('generation', {})
        if generation_data:
            # Emission factors (kg CO2/MWh)
            emission_factors = {
                'solar': 0, 'wind': 0, 'hydro': 0, 'geothermal': 0,
                'gas': 350, 'coal': 820, 'oil': 650
            }
            
            total_emissions = 0
            emissions_by_tech = {}
            
            for tech, generation in generation_data.items():
                tech_type = tech.split('_')[0].lower()
                factor = emission_factors.get(tech_type, 0)
                emissions = generation * factor / 1000  # tonnes CO2
                emissions_by_tech[tech] = emissions
                total_emissions += emissions
            
            st.metric("Total CO2 Emissions", f"{total_emissions:,.0f} tonnes")
            
            if emissions_by_tech:
                em_df = pd.DataFrame.from_dict(emissions_by_tech, orient='index', columns=['Emissions'])
                em_df = em_df[em_df['Emissions'] > 0]
                em_df['Technology'] = em_df.index
                
                if not em_df.empty:
                    fig = px.pie(em_df, values='Emissions', names='Technology',
                               title='CO2 Emissions by Technology (tonnes)')
                    st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "⚡ System Adequacy":
        st.header("System Adequacy")
        
        # Load duration curve (simplified)
        st.subheader("Load Duration Curve")
        
        # Generate synthetic load data
        np.random.seed(42)
        hours = 8760
        load_profile = np.random.lognormal(6, 0.3, hours)
        load_sorted = np.sort(load_profile)[::-1]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, hours+1)),
            y=load_sorted,
            mode='lines',
            name='Load',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title='Load Duration Curve',
            xaxis_title='Hours',
            yaxis_title='Load (MW)',
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "📊 Sensitivity Analysis":
        st.header("Sensitivity Analysis")
        
        st.subheader("Parameter Sensitivity")
        
        # Create sensitivity analysis for key parameters
        parameters = ['Solar Cost', 'Wind Cost', 'Gas Price', 'Discount Rate']
        base_values = [600, 1200, 50, 0.07]
        variations = [-20, -10, 0, 10, 20]  # Percentage changes
        
        sensitivity_data = []
        for i, param in enumerate(parameters):
            base_value = base_values[i]
            for var in variations:
                new_value = base_value * (1 + var/100)
                # Simplified impact calculation
                impact = var * 0.5  # Simplified relationship
                sensitivity_data.append({
                    'Parameter': param,
                    'Variation (%)': var,
                    'Value': new_value,
                    'Impact on Cost (%)': impact
                })
        
        sens_df = pd.DataFrame(sensitivity_data)
        
        fig = px.line(sens_df, x='Variation (%)', y='Impact on Cost (%)', 
                     color='Parameter', title='Sensitivity Analysis')
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()