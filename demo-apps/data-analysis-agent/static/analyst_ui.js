// ClickHouse Data Analyst UI JavaScript - AgentCore Edition

class DataAnalystUI {
    constructor() {
        this.analysisCount = 0;
        this.tables = [];
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadInitialData();
    }

    setupEventListeners() {
        // Tab navigation
        document.querySelectorAll('.tab-button').forEach(button => {
            button.addEventListener('click', (e) => {
                const tabName = e.target.closest('button').dataset.tab;
                this.switchTab(tabName);
            });
        });

        // Analysis controls
        document.getElementById('run-analysis-btn').addEventListener('click', () => this.runTableAnalysis());
        document.getElementById('analysis-type-select').addEventListener('change', () => this.updateColumnSelect());
        document.getElementById('analysis-table-select').addEventListener('change', () => this.loadTableColumns());

        // AI Insights
        document.getElementById('ai-analyze-btn').addEventListener('click', () => this.runAIAnalysis());
        document.getElementById('clear-ai-btn').addEventListener('click', () => this.clearAIQuery());
        document.querySelectorAll('.ai-quick-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const query = e.target.dataset.query;
                document.getElementById('ai-query-input').value = query;
                this.runAIAnalysis();
            });
        });

        // SQL Query Lab
        document.getElementById('execute-sql-btn').addEventListener('click', () => this.executeSQLQuery());
        document.getElementById('clear-sql-btn').addEventListener('click', () => this.clearSQLQuery());
        document.querySelectorAll('.sql-quick-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const query = e.target.dataset.query;
                document.getElementById('sql-query-input').value = query;
                this.executeSQLQuery();
            });
        });

        // Table Comparison
        document.getElementById('run-comparison-btn').addEventListener('click', () => this.runTableComparison());

        // Refresh button
        document.getElementById('refresh-btn').addEventListener('click', () => this.refreshAll());

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                const activeTab = document.querySelector('.tab-content.active').id;
                if (activeTab === 'query') {
                    this.executeSQLQuery();
                } else if (activeTab === 'insights') {
                    this.runAIAnalysis();
                }
            }
        });
    }

    async loadInitialData() {
        try {
            const statusRes = await fetch('/api/status');
            const status = await statusRes.json();
            this.updateConnectionStatus(status);
            
            const tablesRes = await fetch('/api/tables');
            const tablesData = await tablesRes.json();
            
            if (tablesData.success) {
                this.tables = tablesData.tables || [];
                this.updateTablesDisplay();
                this.updateTableSelects();
                document.getElementById('tables-count').textContent = this.tables.length;
            }
            
            document.getElementById('system-status').textContent = 'Ready';
            this.showNotification('Connected to AgentCore', 'success');
        } catch (error) {
            console.error('Failed to load initial data:', error);
            this.updateConnectionStatus({ connected: false, error: error.message });
        }
    }

    updateConnectionStatus(status) {
        const statusEl = document.getElementById('connection-status');
        const iconEl = statusEl.querySelector('i');
        const textEl = statusEl.querySelector('span');
        
        if (status.connected) {
            statusEl.className = 'px-3 py-1 rounded-full text-sm font-medium border bg-green-100 text-green-800 border-green-200';
            iconEl.className = 'fas fa-circle mr-1 text-green-500';
            textEl.textContent = 'AgentCore Ready';
        } else {
            statusEl.className = 'px-3 py-1 rounded-full text-sm font-medium border bg-red-100 text-red-800 border-red-200';
            iconEl.className = 'fas fa-circle mr-1 text-red-500';
            textEl.textContent = 'Error';
            if (status.error) {
                this.showNotification(`Connection error: ${status.error}`, 'error');
            }
        }
    }

    updateTablesDisplay() {
        const tablesList = document.getElementById('tables-list');
        
        if (this.tables.length === 0) {
            tablesList.innerHTML = '<div class="text-center text-gray-500 py-4">No tables found</div>';
            return;
        }
        
        tablesList.innerHTML = '';
        this.tables.forEach(table => {
            const tableItem = document.createElement('div');
            tableItem.className = 'flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors cursor-pointer';
            tableItem.innerHTML = `
                <div class="flex items-center">
                    <i class="fas fa-table text-blue-600 mr-3"></i>
                    <span class="font-medium">${table}</span>
                </div>
                <div class="flex space-x-2">
                    <button class="text-blue-600 hover:text-blue-800 text-sm" onclick="ui.quickAnalyze('${table}')">
                        <i class="fas fa-chart-line"></i>
                    </button>
                    <button class="text-green-600 hover:text-green-800 text-sm" onclick="ui.quickInsights('${table}')">
                        <i class="fas fa-lightbulb"></i>
                    </button>
                </div>
            `;
            tablesList.appendChild(tableItem);
        });
    }

    updateTableSelects() {
        const selects = [
            'analysis-table-select',
            'compare-table1-select', 
            'compare-table2-select'
        ];
        
        selects.forEach(selectId => {
            const select = document.getElementById(selectId);
            select.innerHTML = '<option value="">Select a table...</option>';
            
            this.tables.forEach(table => {
                const option = document.createElement('option');
                option.value = table;
                option.textContent = table;
                select.appendChild(option);
            });
        });
    }


    async loadTableColumns() {
        const tableName = document.getElementById('analysis-table-select').value;
        if (!tableName) return;

        try {
            const res = await fetch(`/api/table/${tableName}/columns`);
            const data = await res.json();
            
            const columnSelect = document.getElementById('analysis-column-select');
            columnSelect.innerHTML = '<option value="">Select a column...</option>';
            
            const commonColumns = ['id', 'name', 'date', 'price', 'quantity', 'category'];
            commonColumns.forEach(column => {
                const option = document.createElement('option');
                option.value = column;
                option.textContent = column;
                option.dataset.isNumeric = ['id', 'price', 'quantity'].includes(column);
                option.dataset.isText = ['name', 'category'].includes(column);
                columnSelect.appendChild(option);
            });
            
            this.updateColumnSelect();
        } catch (error) {
            console.error('Failed to load columns:', error);
        }
    }

    updateColumnSelect() {
        const analysisType = document.getElementById('analysis-type-select').value;
        const columnContainer = document.getElementById('column-select-container');
        
        if (analysisType === 'numeric' || analysisType === 'text') {
            columnContainer.classList.remove('hidden');
            
            const columnSelect = document.getElementById('analysis-column-select');
            const options = columnSelect.querySelectorAll('option');
            
            options.forEach(option => {
                if (option.value === '') return;
                
                const isNumeric = option.dataset.isNumeric === 'true';
                const isText = option.dataset.isText === 'true';
                
                if (analysisType === 'numeric' && !isNumeric) {
                    option.style.display = 'none';
                } else if (analysisType === 'text' && !isText) {
                    option.style.display = 'none';
                } else {
                    option.style.display = 'block';
                }
            });
        } else {
            columnContainer.classList.add('hidden');
        }
    }

    switchTab(tabName) {
        document.querySelectorAll('.tab-button').forEach(btn => {
            btn.classList.remove('active', 'border-blue-500', 'text-blue-600');
            btn.classList.add('border-transparent', 'text-gray-500');
        });
        
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active', 'border-blue-500', 'text-blue-600');
        document.querySelector(`[data-tab="${tabName}"]`).classList.remove('border-transparent', 'text-gray-500');
        
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        
        document.getElementById(tabName).classList.add('active');
    }

    async runTableAnalysis() {
        const tableName = document.getElementById('analysis-table-select').value;
        const analysisType = document.getElementById('analysis-type-select').value;
        const columnName = document.getElementById('analysis-column-select').value;
        
        if (!tableName) {
            this.showNotification('Please select a table', 'warning');
            return;
        }
        
        if ((analysisType === 'numeric' || analysisType === 'text') && !columnName) {
            this.showNotification('Please select a column for this analysis type', 'warning');
            return;
        }
        
        this.showLoading();
        
        try {
            const res = await fetch('/api/table/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ table_name: tableName, analysis_type: analysisType, column_name: columnName })
            });
            
            const data = await res.json();
            
            if (data.success) {
                this.displayAnalysisResult(data.result);
                this.addActivityLog('Table Analysis', `${tableName} - ${analysisType}`);
                this.incrementAnalysisCount();
                this.showNotification('Analysis completed', 'success');
            } else {
                throw new Error(data.detail || 'Analysis failed');
            }
        } catch (error) {
            this.showNotification('Analysis failed: ' + error.message, 'error');
        } finally {
            this.hideLoading();
        }
    }

    async runAIAnalysis() {
        const query = document.getElementById('ai-query-input').value.trim();
        
        if (!query) {
            this.showNotification('Please enter a question', 'warning');
            return;
        }
        
        this.showLoading();
        
        try {
            const res = await fetch('/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query })
            });
            
            const data = await res.json();
            
            if (data.success) {
                this.displayAIResult(data.result);
                this.addActivityLog('AI Analysis', query);
                this.incrementAnalysisCount();
                this.showNotification('AI analysis completed', 'success');
            } else {
                throw new Error(data.detail || 'Analysis failed');
            }
        } catch (error) {
            this.showNotification('AI analysis failed: ' + error.message, 'error');
        } finally {
            this.hideLoading();
        }
    }

    async executeSQLQuery() {
        const query = document.getElementById('sql-query-input').value.trim();
        
        if (!query) {
            this.showNotification('Please enter a SQL query', 'warning');
            return;
        }
        
        this.showLoading();
        
        try {
            const res = await fetch('/api/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query })
            });
            
            const data = await res.json();
            
            if (data.success) {
                this.displaySQLResult(data.result);
                this.addActivityLog('SQL Query', query);
                this.showNotification('Query executed', 'success');
            } else {
                throw new Error(data.detail || 'Query failed');
            }
        } catch (error) {
            this.showNotification('Query failed: ' + error.message, 'error');
        } finally {
            this.hideLoading();
        }
    }

    async runTableComparison() {
        const table1 = document.getElementById('compare-table1-select').value;
        const table2 = document.getElementById('compare-table2-select').value;
        
        if (!table1 || !table2) {
            this.showNotification('Please select both tables', 'warning');
            return;
        }
        
        if (table1 === table2) {
            this.showNotification('Please select different tables', 'warning');
            return;
        }
        
        this.showLoading();
        
        try {
            const res = await fetch('/api/table/compare', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ table1, table2 })
            });
            
            const data = await res.json();
            
            if (data.success) {
                this.displayComparisonResult(data.result);
                this.addActivityLog('Table Comparison', `${table1} vs ${table2}`);
                this.incrementAnalysisCount();
                this.showNotification('Comparison completed', 'success');
            } else {
                throw new Error(data.detail || 'Comparison failed');
            }
        } catch (error) {
            this.showNotification('Comparison failed: ' + error.message, 'error');
        } finally {
            this.hideLoading();
        }
    }


    updateTablesDisplay() {
        const tablesList = document.getElementById('tables-list');
        
        if (this.tables.length === 0) {
            tablesList.innerHTML = '<div class="text-center text-gray-500 py-4">No tables found</div>';
            return;
        }
        
        tablesList.innerHTML = '';
        this.tables.forEach(table => {
            const tableItem = document.createElement('div');
            tableItem.className = 'flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors cursor-pointer';
            tableItem.innerHTML = `
                <div class="flex items-center">
                    <i class="fas fa-table text-blue-600 mr-3"></i>
                    <span class="font-medium">${table}</span>
                </div>
                <div class="flex space-x-2">
                    <button class="text-blue-600 hover:text-blue-800 text-sm" onclick="ui.quickAnalyze('${table}')">
                        <i class="fas fa-chart-line"></i>
                    </button>
                    <button class="text-green-600 hover:text-green-800 text-sm" onclick="ui.quickInsights('${table}')">
                        <i class="fas fa-lightbulb"></i>
                    </button>
                </div>
            `;
            tablesList.appendChild(tableItem);
        });
    }

    updateTableSelects() {
        const selects = ['analysis-table-select', 'compare-table1-select', 'compare-table2-select'];
        
        selects.forEach(selectId => {
            const select = document.getElementById(selectId);
            select.innerHTML = '<option value="">Select a table...</option>';
            
            this.tables.forEach(table => {
                const option = document.createElement('option');
                option.value = table;
                option.textContent = table;
                select.appendChild(option);
            });
        });
    }

    displayAnalysisResult(result) {
        const container = document.getElementById('analysis-results');
        container.innerHTML = `
            <div class="bg-gray-50 rounded-lg p-4">
                <pre class="whitespace-pre-wrap text-sm font-mono overflow-x-auto">${result}</pre>
            </div>
        `;
    }

    displayAIResult(result) {
        const container = document.getElementById('ai-results');
        container.innerHTML = `
            <div class="bg-purple-50 rounded-lg p-4 border-l-4 border-purple-500">
                <div class="flex items-start">
                    <i class="fas fa-robot text-purple-600 text-xl mr-3 mt-1"></i>
                    <div class="flex-1">
                        <h4 class="font-medium text-purple-900 mb-2">AI Analysis Result</h4>
                        <div class="text-gray-800 whitespace-pre-wrap">${result}</div>
                    </div>
                </div>
            </div>
        `;
    }

    displaySQLResult(result) {
        const container = document.getElementById('sql-results');
        container.innerHTML = `
            <div class="bg-green-50 rounded-lg p-4 border-l-4 border-green-500">
                <div class="flex items-start">
                    <i class="fas fa-database text-green-600 text-xl mr-3 mt-1"></i>
                    <div class="flex-1">
                        <h4 class="font-medium text-green-900 mb-2">Query Result</h4>
                        <pre class="whitespace-pre-wrap text-sm font-mono overflow-x-auto text-gray-800">${result}</pre>
                    </div>
                </div>
            </div>
        `;
    }

    displayComparisonResult(result) {
        const container = document.getElementById('comparison-results');
        container.innerHTML = `
            <div class="bg-indigo-50 rounded-lg p-4 border-l-4 border-indigo-500">
                <div class="flex items-start">
                    <i class="fas fa-balance-scale text-indigo-600 text-xl mr-3 mt-1"></i>
                    <div class="flex-1">
                        <h4 class="font-medium text-indigo-900 mb-2">Comparison Result</h4>
                        <pre class="whitespace-pre-wrap text-sm font-mono overflow-x-auto text-gray-800">${result}</pre>
                    </div>
                </div>
            </div>
        `;
    }

    clearAIQuery() {
        document.getElementById('ai-query-input').value = '';
        document.getElementById('ai-results').innerHTML = `
            <div class="text-center text-gray-500 py-12">
                <i class="fas fa-robot text-4xl mb-4"></i>
                <p class="text-lg">Ask the AI analyst a question</p>
                <p class="text-sm">Get intelligent insights about your data using natural language</p>
            </div>
        `;
    }

    clearSQLQuery() {
        document.getElementById('sql-query-input').value = '';
        document.getElementById('sql-results').innerHTML = `
            <div class="text-center text-gray-500 py-12">
                <i class="fas fa-database text-4xl mb-4"></i>
                <p class="text-lg">Execute a query to see results</p>
                <p class="text-sm">Write SQL queries to explore your data</p>
            </div>
        `;
    }

    quickAnalyze(tableName) {
        this.switchTab('analysis');
        document.getElementById('analysis-table-select').value = tableName;
        document.getElementById('analysis-type-select').value = 'full';
        this.loadTableColumns();
        setTimeout(() => this.runTableAnalysis(), 500);
    }

    quickInsights(tableName) {
        this.switchTab('insights');
        document.getElementById('ai-query-input').value = `Provide comprehensive insights and analysis for the ${tableName} table`;
        setTimeout(() => this.runAIAnalysis(), 500);
    }

    addActivityLog(type, description) {
        const activityLog = document.getElementById('activity-log');
        const timestamp = new Date().toLocaleTimeString();
        
        const logItem = document.createElement('div');
        logItem.className = 'flex items-center justify-between p-3 bg-gray-50 rounded-lg';
        logItem.innerHTML = `
            <div class="flex items-center">
                <i class="fas fa-clock text-gray-400 mr-3"></i>
                <div>
                    <p class="font-medium text-sm">${type}</p>
                    <p class="text-xs text-gray-600">${description}</p>
                </div>
            </div>
            <span class="text-xs text-gray-500">${timestamp}</span>
        `;
        
        const placeholder = activityLog.querySelector('.text-center');
        if (placeholder) {
            placeholder.remove();
        }
        
        activityLog.insertBefore(logItem, activityLog.firstChild);
        
        while (activityLog.children.length > 10) {
            activityLog.removeChild(activityLog.lastChild);
        }
    }

    incrementAnalysisCount() {
        this.analysisCount++;
        document.getElementById('analysis-count').textContent = this.analysisCount;
    }

    showLoading() {
        document.getElementById('loading-overlay').classList.remove('hidden');
        document.getElementById('loading-overlay').classList.add('flex');
    }

    hideLoading() {
        document.getElementById('loading-overlay').classList.add('hidden');
        document.getElementById('loading-overlay').classList.remove('flex');
    }

    showNotification(message, type = 'info') {
        const container = document.getElementById('notifications');
        const notification = document.createElement('div');
        
        const colors = {
            success: 'bg-green-100 border-green-400 text-green-700',
            error: 'bg-red-100 border-red-400 text-red-700',
            warning: 'bg-yellow-100 border-yellow-400 text-yellow-700',
            info: 'bg-blue-100 border-blue-400 text-blue-700'
        };
        
        const icons = {
            success: 'fas fa-check-circle',
            error: 'fas fa-exclamation-circle',
            warning: 'fas fa-exclamation-triangle',
            info: 'fas fa-info-circle'
        };
        
        notification.className = `notification border-l-4 p-4 rounded-lg shadow-lg max-w-sm ${colors[type]}`;
        notification.innerHTML = `
            <div class="flex">
                <i class="${icons[type]} mr-3 mt-0.5"></i>
                <div class="flex-1">
                    <p class="text-sm font-medium">${message}</p>
                </div>
                <button class="ml-3 text-lg leading-none" onclick="this.parentElement.parentElement.remove()">×</button>
            </div>
        `;
        
        container.appendChild(notification);
        
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, 5000);
    }

    refreshAll() {
        this.loadInitialData();
        this.showNotification('Data refreshed', 'success');
    }
}

const ui = new DataAnalystUI();