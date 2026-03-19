class ScreenMonitor {
    constructor() {
        this.statusUpdateInterval = null;
        this.imageRefreshInterval = null;
        this.isCapturing = false;
        
        this.initializeEventListeners();
        this.updateStatus();
        this.loadWebhooks();
        this.startStatusUpdates();
    }
    
    initializeEventListeners() {
        // Control buttons
        document.getElementById('start-btn').addEventListener('click', () => this.startCapture());
        document.getElementById('stop-btn').addEventListener('click', () => this.stopCapture());
        document.getElementById('update-config-btn').addEventListener('click', () => this.updateConfig());
        document.getElementById('refresh-image-btn').addEventListener('click', () => this.refreshImage());
        
        // Webhook management
        document.getElementById('add-webhook-btn').addEventListener('click', () => this.addWebhook());
        document.getElementById('test-webhook-btn').addEventListener('click', () => this.testWebhook());
        document.getElementById('enable-external').addEventListener('change', () => this.toggleExternalSending());
        document.getElementById('external-format').addEventListener('change', () => this.updateExternalFormat());
        
        // Form validation
        const form = document.getElementById('control-form');
        form.addEventListener('input', this.validateForm.bind(this));
    }
    
    async startCapture() {
        try {
            const config = this.getFormConfig();
            
            const response = await fetch('/api/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(config)
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showMessage('Screen capture started successfully!', 'success');
                this.isCapturing = true;
                this.updateControlButtons();
                this.startImageRefresh();
            } else {
                this.showMessage(`Failed to start capture: ${result.message}`, 'danger');
            }
        } catch (error) {
            this.showMessage(`Error starting capture: ${error.message}`, 'danger');
        }
    }
    
    async stopCapture() {
        try {
            const response = await fetch('/api/stop', {
                method: 'POST'
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showMessage('Screen capture stopped!', 'info');
                this.isCapturing = false;
                this.updateControlButtons();
                this.stopImageRefresh();
            } else {
                this.showMessage(`Failed to stop capture: ${result.message}`, 'danger');
            }
        } catch (error) {
            this.showMessage(`Error stopping capture: ${error.message}`, 'danger');
        }
    }
    
    async updateConfig() {
        try {
            const config = this.getFormConfig();
            
            const response = await fetch('/api/config', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(config)
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showMessage('Configuration updated successfully!', 'success');
                this.updateConfigDisplay(result.config);
            } else {
                this.showMessage(`Failed to update configuration: ${result.message}`, 'warning');
            }
        } catch (error) {
            this.showMessage(`Error updating configuration: ${error.message}`, 'danger');
        }
    }
    
    async refreshImage() {
        try {
            const response = await fetch('/api/image/latest');
            const result = await response.json();
            
            if (result.success) {
                this.displayImage(result);
            } else {
                this.showMessage(`No image available: ${result.message}`, 'info');
            }
        } catch (error) {
            this.showMessage(`Error refreshing image: ${error.message}`, 'danger');
        }
    }
    
    async updateStatus() {
        try {
            const response = await fetch('/api/status');
            const status = await response.json();
            
            this.isCapturing = status.is_capturing;
            this.updateStatusDisplay(status);
            this.updateControlButtons();
            
            // Show error if any
            if (status.last_error) {
                this.showMessage(`Last error: ${status.last_error}`, 'warning');
            }
        } catch (error) {
            console.error('Error updating status:', error);
        }
    }
    
    updateStatusDisplay(status) {
        // Update status indicator
        const statusIndicator = document.getElementById('status-indicator');
        const captureStatus = document.getElementById('capture-status');
        
        if (status.is_capturing) {
            statusIndicator.innerHTML = '<i data-feather="circle" class="me-1" style="width: 12px; height: 12px;"></i> Capturing';
            statusIndicator.className = 'badge bg-success';
            captureStatus.innerHTML = '<i data-feather="play" class="me-1" style="width: 12px; height: 12px;"></i> Capturing';
            captureStatus.className = 'badge bg-success';
        } else {
            statusIndicator.innerHTML = '<i data-feather="circle" class="me-1" style="width: 12px; height: 12px;"></i> Stopped';
            statusIndicator.className = 'badge bg-secondary';
            captureStatus.innerHTML = '<i data-feather="pause" class="me-1" style="width: 12px; height: 12px;"></i> Stopped';
            captureStatus.className = 'badge bg-secondary';
        }
        
        // Update config display
        if (status.config) {
            document.getElementById('current-interval').textContent = `${status.config.interval}s`;
            document.getElementById('current-quality').textContent = `${status.config.quality}%`;
        }
        
        // Update statistics
        if (status.stats) {
            document.getElementById('total-captures').textContent = status.stats.total_captures || 0;
            document.getElementById('success-count').textContent = (status.stats.total_captures || 0) - (status.stats.failed_captures || 0);
            document.getElementById('failed-count').textContent = status.stats.failed_captures || 0;
            
            // Update uptime
            if (status.stats.uptime_seconds) {
                const uptime = this.formatUptime(status.stats.uptime_seconds);
                document.getElementById('uptime').textContent = uptime;
            }
        }
        
        // Re-render feather icons
        feather.replace();
    }
    
    updateControlButtons() {
        const startBtn = document.getElementById('start-btn');
        const stopBtn = document.getElementById('stop-btn');
        
        startBtn.disabled = this.isCapturing;
        stopBtn.disabled = !this.isCapturing;
    }
    
    updateConfigDisplay(config) {
        document.getElementById('current-interval').textContent = `${config.interval}s`;
        document.getElementById('current-quality').textContent = `${config.quality}%`;
    }
    
    displayImage(imageData) {
        const imageContainer = document.getElementById('image-container');
        const placeholder = document.getElementById('no-image-placeholder');
        const image = document.getElementById('latest-image');
        const imageInfo = document.getElementById('image-info');
        
        // Hide placeholder, show image
        placeholder.classList.add('d-none');
        image.classList.remove('d-none');
        imageInfo.classList.remove('d-none');
        
        // Set image source
        image.src = imageData.image;
        
        // Update image info
        document.getElementById('image-size').textContent = `${imageData.size[0]} x ${imageData.size[1]}`;
        document.getElementById('image-timestamp').textContent = new Date(imageData.timestamp).toLocaleString();
        document.getElementById('last-capture-time').textContent = new Date(imageData.timestamp).toLocaleTimeString();
    }
    
    getFormConfig() {
        return {
            interval: parseFloat(document.getElementById('interval').value),
            quality: parseInt(document.getElementById('quality').value),
            resize_factor: parseFloat(document.getElementById('resize-factor').value),
            add_timestamp: document.getElementById('add-timestamp').checked,
            save_local: document.getElementById('save-local').checked,
            save_path: document.getElementById('save-path').value.trim()
        };
    }
    
    validateForm() {
        const interval = parseFloat(document.getElementById('interval').value);
        const quality = parseInt(document.getElementById('quality').value);
        const resizeFactor = parseFloat(document.getElementById('resize-factor').value);
        const saveLocal = document.getElementById('save-local').checked;
        const savePath = document.getElementById('save-path').value.trim();
        
        let isValid = true;
        
        // Validate interval
        if (interval <= 0 || interval > 60) {
            isValid = false;
        }
        
        // Validate quality
        if (quality < 1 || quality > 100) {
            isValid = false;
        }
        
        // Validate resize factor
        if (resizeFactor <= 0 || resizeFactor > 2) {
            isValid = false;
        }

        // Validate save path when local saving is enabled
        if (saveLocal && !savePath) {
            isValid = false;
        }
        
        // Update button states
        document.getElementById('update-config-btn').disabled = !isValid;
        
        return isValid;
    }
    
    showMessage(message, type = 'info') {
        const container = document.getElementById('error-container');
        
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        container.appendChild(alertDiv);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }
    
    startStatusUpdates() {
        this.statusUpdateInterval = setInterval(() => {
            this.updateStatus();
        }, 2000); // Update every 2 seconds
    }
    
    startImageRefresh() {
        this.imageRefreshInterval = setInterval(() => {
            if (this.isCapturing) {
                this.refreshImage();
            }
        }, 3000); // Refresh image every 3 seconds
    }
    
    stopImageRefresh() {
        if (this.imageRefreshInterval) {
            clearInterval(this.imageRefreshInterval);
            this.imageRefreshInterval = null;
        }
    }
    
    formatUptime(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        
        return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    
    async loadWebhooks() {
        try {
            const response = await fetch('/api/webhooks');
            const result = await response.json();
            
            if (result.success) {
                this.displayWebhooks(result.webhooks);
                document.getElementById('enable-external').checked = result.external_sending_enabled;
                document.getElementById('external-format').value = result.external_format;
                this.updateWebhookStatus(result.external_sending_enabled);
            }
        } catch (error) {
            console.error('Error loading webhooks:', error);
        }
    }
    
    displayWebhooks(webhooks) {
        const container = document.getElementById('webhook-list');
        const noWebhooksMsg = document.getElementById('no-webhooks-msg');
        
        if (webhooks.length === 0) {
            noWebhooksMsg.style.display = 'block';
            return;
        }
        
        noWebhooksMsg.style.display = 'none';
        
        // Clear existing webhooks except the no-webhooks message
        Array.from(container.children).forEach(child => {
            if (child.id !== 'no-webhooks-msg') {
                child.remove();
            }
        });
        
        webhooks.forEach(url => {
            const webhookDiv = document.createElement('div');
            webhookDiv.className = 'webhook-item d-flex justify-content-between align-items-center mb-2 p-2 bg-body rounded';
            webhookDiv.innerHTML = `
                <span class="text-truncate me-2">${url}</span>
                <button class="btn btn-sm btn-outline-danger remove-webhook-btn" data-url="${url}">
                    <i data-feather="trash-2" style="width: 14px; height: 14px;"></i>
                </button>
            `;
            
            container.appendChild(webhookDiv);
            
            // Add remove event listener
            const removeBtn = webhookDiv.querySelector('.remove-webhook-btn');
            removeBtn.addEventListener('click', () => this.removeWebhook(url));
        });
        
        // Re-render feather icons
        feather.replace();
    }
    
    async addWebhook() {
        const urlInput = document.getElementById('webhook-url');
        const url = urlInput.value.trim();
        
        if (!url) {
            this.showMessage('Please enter a webhook URL', 'warning');
            return;
        }
        
        try {
            const response = await fetch('/api/webhooks', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ add_url: url })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showMessage(result.message, 'success');
                urlInput.value = '';
                this.displayWebhooks(result.webhooks);
            } else {
                this.showMessage(result.message, 'danger');
            }
        } catch (error) {
            this.showMessage(`Error adding webhook: ${error.message}`, 'danger');
        }
    }
    
    async removeWebhook(url) {
        try {
            const response = await fetch('/api/webhooks', {
                method: 'DELETE',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ remove_url: url })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showMessage(result.message, 'info');
                this.displayWebhooks(result.webhooks);
            } else {
                this.showMessage(result.message, 'danger');
            }
        } catch (error) {
            this.showMessage(`Error removing webhook: ${error.message}`, 'danger');
        }
    }
    
    async testWebhook() {
        const urlInput = document.getElementById('webhook-url');
        const url = urlInput.value.trim();
        
        if (!url) {
            this.showMessage('Please enter a webhook URL to test', 'warning');
            return;
        }
        
        try {
            const response = await fetch('/api/test-webhook', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ url: url })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showMessage(result.message, 'success');
            } else {
                this.showMessage(result.message, 'danger');
            }
        } catch (error) {
            this.showMessage(`Error testing webhook: ${error.message}`, 'danger');
        }
    }
    
    async toggleExternalSending() {
        const enabled = document.getElementById('enable-external').checked;
        
        try {
            const response = await fetch('/api/webhooks', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ enable_external: enabled })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showMessage(result.message, 'success');
                this.updateWebhookStatus(enabled);
            } else {
                this.showMessage(result.message, 'danger');
            }
        } catch (error) {
            this.showMessage(`Error toggling external sending: ${error.message}`, 'danger');
        }
    }
    
    async updateExternalFormat() {
        const format = document.getElementById('external-format').value;
        
        try {
            const response = await fetch('/api/webhooks', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ external_format: format })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showMessage(result.message, 'success');
            } else {
                this.showMessage(result.message, 'danger');
            }
        } catch (error) {
            this.showMessage(`Error updating format: ${error.message}`, 'danger');
        }
    }
    
    updateWebhookStatus(enabled) {
        const statusElement = document.getElementById('webhook-status');
        if (enabled) {
            statusElement.textContent = 'Active';
            statusElement.className = 'fw-bold text-success';
        } else {
            statusElement.textContent = 'Disabled';
            statusElement.className = 'fw-bold text-secondary';
        }
    }
}

// Initialize the monitor when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new ScreenMonitor();
});
