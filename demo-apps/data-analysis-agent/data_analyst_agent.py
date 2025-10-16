#!/usr/bin/env python3
"""
ClickHouse Data Analyst Agent
Specialized agent for data analysis and summarization using ClickHouse MCP server.
"""

from strands import Agent, tool
from simple_mcp_agent import create_clickhouse_agent


class ClickHouseDataAnalyst(Agent):
    """Specialized agent for ClickHouse data analysis and summarization."""
    
    def __init__(self):
        # Create MCP client for ClickHouse
        self.mcp_client = create_clickhouse_agent()
        self.mcp_context_active = False
        
        super().__init__(
            name="ClickHouse Data Analyst",
            description="Expert data analyst specializing in ClickHouse database analysis and insights",
            system_prompt="""You are an expert ClickHouse data analyst with deep knowledge of:

🔍 DATA ANALYSIS CAPABILITIES:
- Statistical analysis of numeric data (mean, median, quartiles, distributions)
- Text data analysis (frequency, patterns, uniqueness)
- Data quality assessment (completeness, duplicates, anomalies)
- Performance analysis and optimization recommendations
- Comparative analysis between datasets
- Trend identification and pattern recognition

📊 ANALYSIS APPROACH:
1. Always start with basic data exploration (row counts, schema, sample data)
2. Identify data types and appropriate analysis methods
3. Generate comprehensive statistical summaries
4. Look for patterns, outliers, and data quality issues
5. Provide actionable insights and recommendations
6. Format results clearly with visual indicators and explanations

💡 COMMUNICATION STYLE:
- Use clear, business-friendly language
- Provide context for technical metrics
- Highlight key findings and actionable insights
- Include data quality observations
- Suggest next steps for further analysis

When users ask for data analysis:
- Use the appropriate analysis tools based on data types
- Provide comprehensive summaries with key metrics
- Identify interesting patterns or anomalies
- Make recommendations for data improvement or further investigation""",
            tools=[]
        )
    
    def setup_mcp_tools(self):
        """Setup MCP tools within context manager."""
        if not self.mcp_context_active:
            self.mcp_client.__enter__()
            self.mcp_context_active = True
            
            # Get MCP tools and add them using the inherited tool registry
            tools = self.mcp_client.list_tools_sync()
            
            # Use the inherited tool registry to process the tools
            self.tool_registry.process_tools(tools)
    
    def cleanup(self):
        """Cleanup MCP context."""
        if self.mcp_context_active and self.mcp_client:
            self.mcp_client.__exit__(None, None, None)
            self.mcp_context_active = False
    
    @tool
    def analyze_data_trends(self, table_name: str, date_column: str = None, metric_column: str = None) -> str:
        """
        Analyze trends in data over time.
        
        Args:
            table_name: Name of the table to analyze
            date_column: Name of the date/timestamp column (optional)
            metric_column: Name of the metric column to analyze (optional)
            
        Returns:
            Trend analysis summary
        """
        try:
            # If no date column specified, try to find one
            if not date_column:
                date_columns_query = f"""
                SELECT name 
                FROM system.columns 
                WHERE table = '{table_name}' 
                AND database = currentDatabase()
                AND (type LIKE '%Date%' OR type LIKE '%DateTime%')
                LIMIT 1
                """
                date_result = self.clickhouse_agent.execute_query(date_columns_query)
                if "Error" not in date_result and date_result.strip():
                    # Extract column name from result
                    lines = date_result.split('\n')
                    for line in lines:
                        if '|' in line and 'name' not in line and '---' not in line:
                            date_column = line.split('|')[0].strip()
                            break
            
            if not date_column:
                return f"No date/timestamp column found in table '{table_name}'. Please specify a date_column parameter."
            
            # Analyze trends by day/week/month
            trends_query = f"""
            SELECT 
                toDate({date_column}) as date,
                count() as daily_count,
                countDistinct(*) as unique_records
            FROM {table_name}
            WHERE {date_column} IS NOT NULL
            GROUP BY toDate({date_column})
            ORDER BY date DESC
            LIMIT 30
            """
            
            trends_result = self.clickhouse_agent.execute_query(trends_query)
            
            # Monthly aggregation
            monthly_query = f"""
            SELECT 
                toYYYYMM({date_column}) as year_month,
                count() as monthly_count,
                min(toDate({date_column})) as period_start,
                max(toDate({date_column})) as period_end
            FROM {table_name}
            WHERE {date_column} IS NOT NULL
            GROUP BY toYYYYMM({date_column})
            ORDER BY year_month DESC
            LIMIT 12
            """
            
            monthly_result = self.clickhouse_agent.execute_query(monthly_query)
            
            analysis = f"📈 TREND ANALYSIS FOR {table_name}\n"
            analysis += "=" * 40 + "\n\n"
            
            analysis += f"📅 DAILY TRENDS (Last 30 days):\n"
            analysis += f"{trends_result}\n\n"
            
            analysis += f"📊 MONTHLY TRENDS (Last 12 months):\n"
            analysis += f"{monthly_result}\n\n"
            
            # If metric column specified, analyze its trends
            if metric_column:
                metric_trends_query = f"""
                SELECT 
                    toDate({date_column}) as date,
                    count() as record_count,
                    avg({metric_column}) as avg_value,
                    min({metric_column}) as min_value,
                    max({metric_column}) as max_value,
                    sum({metric_column}) as total_value
                FROM {table_name}
                WHERE {date_column} IS NOT NULL AND {metric_column} IS NOT NULL
                GROUP BY toDate({date_column})
                ORDER BY date DESC
                LIMIT 14
                """
                
                metric_result = self.clickhouse_agent.execute_query(metric_trends_query)
                analysis += f"💰 METRIC TRENDS ({metric_column} - Last 14 days):\n"
                analysis += f"{metric_result}\n\n"
            
            return analysis
            
        except Exception as e:
            return f"Error analyzing trends for table '{table_name}': {str(e)}"
    
    @tool
    def detect_data_anomalies(self, table_name: str, column_name: str = None) -> str:
        """
        Detect anomalies and outliers in data.
        
        Args:
            table_name: Name of the table to analyze
            column_name: Specific column to analyze (optional)
            
        Returns:
            Anomaly detection summary
        """
        try:
            # If no column specified, find numeric columns
            if not column_name:
                numeric_columns_query = f"""
                SELECT name 
                FROM system.columns 
                WHERE table = '{table_name}' 
                AND database = currentDatabase()
                AND (type LIKE '%Int%' OR type LIKE '%Float%' OR type LIKE '%Decimal%')
                LIMIT 1
                """
                numeric_result = self.clickhouse_agent.execute_query(numeric_columns_query)
                if "Error" not in numeric_result:
                    lines = numeric_result.split('\n')
                    for line in lines:
                        if '|' in line and 'name' not in line and '---' not in line:
                            column_name = line.split('|')[0].strip()
                            break
            
            if not column_name:
                return f"No numeric column specified or found in table '{table_name}'"
            
            # Statistical analysis for outlier detection
            stats_query = f"""
            SELECT 
                count() as total_count,
                avg({column_name}) as mean_value,
                stddevPop({column_name}) as std_dev,
                quantile(0.25)({column_name}) as q1,
                quantile(0.75)({column_name}) as q3,
                quantile(0.75)({column_name}) - quantile(0.25)({column_name}) as iqr,
                min({column_name}) as min_value,
                max({column_name}) as max_value
            FROM {table_name}
            WHERE {column_name} IS NOT NULL
            """
            
            stats_result = self.clickhouse_agent.execute_query(stats_query)
            
            # Find potential outliers using IQR method
            outliers_query = f"""
            WITH stats AS (
                SELECT 
                    quantile(0.25)({column_name}) as q1,
                    quantile(0.75)({column_name}) as q3,
                    quantile(0.75)({column_name}) - quantile(0.25)({column_name}) as iqr
                FROM {table_name}
                WHERE {column_name} IS NOT NULL
            )
            SELECT 
                'Outliers Analysis' as analysis,
                count() as potential_outliers,
                min({column_name}) as min_outlier,
                max({column_name}) as max_outlier
            FROM {table_name}, stats
            WHERE {column_name} < (q1 - 1.5 * iqr) OR {column_name} > (q3 + 1.5 * iqr)
            """
            
            outliers_result = self.clickhouse_agent.execute_query(outliers_query)
            
            # Check for duplicate records
            duplicates_query = f"""
            SELECT 
                'Duplicate Analysis' as analysis,
                count() as total_records,
                count() - count(DISTINCT *) as duplicate_records,
                (count() - count(DISTINCT *)) * 100.0 / count() as duplicate_percentage
            FROM {table_name}
            """
            
            duplicates_result = self.clickhouse_agent.execute_query(duplicates_query)
            
            anomalies = f"🚨 ANOMALY DETECTION FOR {table_name}.{column_name}\n"
            anomalies += "=" * 50 + "\n\n"
            
            anomalies += "📊 STATISTICAL SUMMARY:\n"
            anomalies += f"{stats_result}\n\n"
            
            anomalies += "⚠️ OUTLIER ANALYSIS:\n"
            anomalies += f"{outliers_result}\n\n"
            
            anomalies += "🔍 DUPLICATE ANALYSIS:\n"
            anomalies += f"{duplicates_result}\n\n"
            
            anomalies += "💡 INTERPRETATION:\n"
            anomalies += "- Values beyond Q1-1.5*IQR or Q3+1.5*IQR are potential outliers\n"
            anomalies += "- High duplicate percentage may indicate data quality issues\n"
            anomalies += "- Large standard deviation suggests high variability\n"
            
            return anomalies
            
        except Exception as e:
            return f"Error detecting anomalies in '{table_name}.{column_name}': {str(e)}"
    
    @tool
    def generate_executive_summary(self, table_name: str) -> str:
        """
        Generate an executive summary of a dataset.
        
        Args:
            table_name: Name of the table to summarize
            
        Returns:
            Executive summary with key insights
        """
        try:
            # Get comprehensive analysis
            basic_analysis = self.clickhouse_agent.analyze_table_data(table_name)
            insights = self.clickhouse_agent.generate_data_insights(table_name)
            
            # Get key metrics
            key_metrics_query = f"""
            SELECT 
                count() as total_records,
                count(DISTINCT *) as unique_records,
                (SELECT count() FROM system.columns WHERE table = '{table_name}' AND database = currentDatabase()) as total_columns,
                formatReadableSize((SELECT sum(bytes) FROM system.parts WHERE table = '{table_name}' AND active)) as table_size
            FROM {table_name}
            """
            
            metrics_result = self.clickhouse_agent.execute_query(key_metrics_query)
            
            # Data freshness
            freshness_query = f"""
            SELECT 
                'Data Freshness' as metric,
                max(modification_time) as last_modified
            FROM system.parts 
            WHERE table = '{table_name}' AND active
            """
            
            freshness_result = self.clickhouse_agent.execute_query(freshness_query)
            
            summary = f"📋 EXECUTIVE SUMMARY: {table_name.upper()}\n"
            summary += "=" * 50 + "\n\n"
            
            summary += "🎯 KEY METRICS:\n"
            summary += f"{metrics_result}\n\n"
            
            summary += "⏰ DATA FRESHNESS:\n"
            summary += f"{freshness_result}\n\n"
            
            summary += "🔍 QUICK INSIGHTS:\n"
            # Extract key points from insights
            if "OVERVIEW:" in insights:
                overview_section = insights.split("OVERVIEW:")[1].split("STRUCTURE:")[0]
                summary += f"{overview_section}\n"
            
            summary += "📊 RECOMMENDATIONS:\n"
            summary += "• Regular data quality monitoring\n"
            summary += "• Performance optimization review\n"
            summary += "• Data governance implementation\n"
            summary += "• Automated anomaly detection setup\n\n"
            
            summary += "📈 NEXT STEPS:\n"
            summary += "• Detailed column-level analysis\n"
            summary += "• Trend analysis over time\n"
            summary += "• Cross-table relationship analysis\n"
            summary += "• Business impact assessment\n"
            
            return summary
            
        except Exception as e:
            return f"Error generating executive summary for '{table_name}': {str(e)}"


def main():
    """Demo of the data analyst agent."""
    
    print("🔍 ClickHouse Data Analyst Agent Demo")
    print("=" * 40)
    
    # Create the analyst agent
    analyst = ClickHouseDataAnalyst()
    
    # Example analysis queries
    analysis_tasks = [
        "What tables are available in the database?",
        "Analyze the structure and content of the system.tables table",
        "Generate an executive summary for the system.databases table",
        "What are the key insights about data in the system.columns table?",
        "Detect any anomalies in the system.parts table",
        "Compare the system.tables and system.columns tables"
    ]
    
    try:
        for i, task in enumerate(analysis_tasks, 1):
            print(f"\n🔍 Analysis Task {i}: {task}")
            print("-" * 60)
            
            response = analyst(task)
            
            # Show truncated response for demo
            if len(response) > 500:
                print(f"{response[:500]}...")
                print(f"[Response truncated - {len(response)} total characters]")
            else:
                print(response)
            
            print("\n" + "="*60)
        
        print("\n✅ Data analysis demo completed!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")


if __name__ == "__main__":
    main()