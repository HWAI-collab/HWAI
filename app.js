// HWAI Academic Portfolio - Repository Display with Local Data
let allRepos = [];
let readmeCache = {};

// Technology icon mapping using Devicon CDN
function getTechIcon(tech) {
    const techIcons = {
        // Languages
        'Vue.js': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/vuejs/vuejs-original.svg',
        'Vue.js 3': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/vuejs/vuejs-original.svg',
        'Vue': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/vuejs/vuejs-original.svg',
        'JavaScript': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/javascript/javascript-original.svg',
        'TypeScript': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/typescript/typescript-original.svg',
        'Python': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg',
        'Python 3.8+': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg',
        'Dart': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/dart/dart-original.svg',
        'Dart 3.0+': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/dart/dart-original.svg',
        'Dart SDK': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/dart/dart-original.svg',
        'R': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/r/r-original.svg',
        
        // Frameworks
        'Nuxt 3': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/nuxtjs/nuxtjs-original.svg',
        'Nuxt': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/nuxtjs/nuxtjs-original.svg',
        'Flutter': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/flutter/flutter-original.svg',
        'Flutter Web': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/flutter/flutter-original.svg',
        'Flutter 3.5.4+': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/flutter/flutter-original.svg',
        'Flutter 3.4+': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/flutter/flutter-original.svg',
        'Django': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/django/django-plain.svg',
        'Django 4.2.5': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/django/django-plain.svg',
        'Node.js': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/nodejs/nodejs-original.svg',
        
        // Databases
        'PostgreSQL': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/postgresql/postgresql-original.svg',
        'Supabase': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/supabase/supabase-original.svg',
        'Firebase': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/firebase/firebase-plain.svg',
        'Hive': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/apache/apache-original.svg',
        
        // ML/AI Libraries
        'TensorFlow': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/tensorflow/tensorflow-original.svg',
        'PyTorch': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pytorch/pytorch-original.svg',
        'PyTorch 2.0+': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pytorch/pytorch-original.svg',
        'OpenCV': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/opencv/opencv-original.svg',
        'OpenCV 4.8+': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/opencv/opencv-original.svg',
        'Scikit-learn': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/scikitlearn/scikitlearn-original.svg',
        'Pandas': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pandas/pandas-original.svg',
        'NumPy': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/numpy/numpy-original.svg',
        
        // Charts/Visualization
        'Chart.js': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/chartjs/chartjs-original.svg',
        'ggplot2': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/r/r-original.svg',
        'FL Chart': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/flutter/flutter-original.svg',
        
        // Payment/APIs
        'Stripe': 'https://images.stripeassets.com/fzn2n1nzq965/HTTOloNPhisV9P4hlMPNA/cacf1bb88b9fc492dfad34378d844280/Stripe_icon_-_square.svg?q=80&w=1082',
        'Google APIs': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/google/google-original.svg',
        
        // Generic categories
        'AI/ML': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/tensorflow/tensorflow-original.svg',
        'ML': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/tensorflow/tensorflow-original.svg',
        'NLP': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg',
        'PWA': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/chrome/chrome-original.svg',
        'OCR': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/opencv/opencv-original.svg',
        'WebRTC': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/chrome/chrome-original.svg',
        'AI APIs': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg',
        'Analytics APIs': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/chartjs/chartjs-original.svg',
        'Dashboard': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/vuejs/vuejs-original.svg',
        'PostGIS': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/postgresql/postgresql-original.svg',
        'Mapbox': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/google/google-original.svg',
        'BeautifulSoup': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg',
        'GeoPandas': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pandas/pandas-original.svg',
        'Bioconductor': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/r/r-original.svg',
        'phyloseq': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/r/r-original.svg',
        'ape': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/r/r-original.svg',
        'YOLOv8': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg',
        'GANs': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/tensorflow/tensorflow-original.svg',
        'ONNX': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg',
        'Ultralytics': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg',
        'DeepSeek AI': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg',
        'pymem': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg',
        'httpx': 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg'
    };
    
    return techIcons[tech] || 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/devicon/devicon-original.svg';
}

// Initialize the application
document.addEventListener('DOMContentLoaded', init);

async function init() {
    await loadLocalRepositories();
    setupEventListeners();
    setupAccessRequestModal();
}

// Setup event listeners for search and filter
function setupEventListeners() {
    const searchInput = document.getElementById('searchInput');
    const filterCategory = document.getElementById('filterCategory');
    
    searchInput.addEventListener('input', filterRepos);
    filterCategory.addEventListener('change', filterRepos);
}

// Load repositories from local JSON file
async function loadLocalRepositories() {
    const loading = document.getElementById('loading');
    const errorMessage = document.getElementById('errorMessage');
    const repoGrid = document.getElementById('repoGrid');
    
    try {
        // Fetch local repository data
        const response = await fetch('data/repositories.json');
        
        if (!response.ok) {
            throw new Error('Failed to load repository data');
        }
        
        const data = await response.json();
        const repos = data.repositories;
        
        console.log('Loaded repositories from JSON:', repos.length); // Debug log
        
        // Process each repository
        for (const repo of repos) {
            const repoData = {
                name: repo.name,
                title: repo.title,
                description: repo.description,
                url: repo.url,
                stars: repo.stars,
                forks: repo.forks,
                language: repo.language,
                updated: new Date(),
                topics: repo.topics || [],
                category: repo.category,
                abstract: repo.abstract,
                sections: repo.sections,
                highlights: repo.highlights,
                technologies: repo.technologies,
                readme_content: repo.readme_content
            };
            
            allRepos.push(repoData);
            
            // Store README content in cache
            readmeCache[repo.name] = {
                title: repo.title,
                abstract: repo.abstract,
                sections: repo.sections,
                highlights: repo.highlights,
                technologies: repo.technologies
            };
        }
        
        // Sort repositories by stars (descending)
        allRepos.sort((a, b) => b.stars - a.stars);
        
        // Display repositories
        displayRepos(allRepos);
        loading.style.display = 'none';
        
    } catch (error) {
        console.error('Error loading repositories:', error);
        loading.style.display = 'none';
        errorMessage.style.display = 'block';
        errorMessage.innerHTML = '<p>Unable to load repository data. Please refresh the page to try again.</p>';
    }
}

// Display repositories in the grid
function displayRepos(repos) {
    const repoGrid = document.getElementById('repoGrid');
    repoGrid.innerHTML = '';
    
    console.log('Displaying repositories:', repos.length); // Debug log
    
    if (repos.length === 0) {
        repoGrid.innerHTML = '<p class="no-results">No repositories found matching your criteria.</p>';
        return;
    }
    
    repos.forEach(repo => {
        const readme = readmeCache[repo.name] || {};
        const card = createRepoCard(repo, readme);
        repoGrid.appendChild(card);
    });
}

// Create a repository card element
function createRepoCard(repo, readme) {
    const card = document.createElement('article');
    card.className = 'repo-card';
    card.dataset.category = repo.category;
    
    const categoryClass = `category-${repo.category}`;
    card.classList.add(categoryClass);
    
    card.innerHTML = `
        <div class="card-header">
            <h2>${readme.title || repo.name}</h2>
            <span class="category-badge">${repo.category.toUpperCase()}</span>
        </div>
        
        <div class="card-abstract">
            <p>${readme.abstract || repo.description}</p>
        </div>
        
        ${readme.highlights && readme.highlights.length > 0 ? `
        <div class="card-sections">
            <h3>Key Features</h3>
            <ul>
                ${readme.highlights.slice(0, 5).map(highlight => `<li>${highlight}</li>`).join('')}
            </ul>
        </div>
        ` : ''}
        
        ${readme.technologies && readme.technologies.length > 0 ? `
        <div class="technologies">
            ${readme.technologies.slice(0, 5).map(tech => 
                `<span class="tech-badge"><img src="${getTechIcon(tech)}" alt="${tech}" class="tech-icon"> ${tech}</span>`
            ).join('')}
        </div>
        ` : ''}
        
        <div class="card-footer">
            <button class="btn-primary" onclick="requestAccess('${repo.name}')">Request Source Code</button>
            <button class="btn-primary" onclick="viewReadmePopup('${repo.name}')">View README</button>
        </div>
    `;
    
    
    return card;
}

// Filter repositories based on search and category
function filterRepos() {
    const searchTerm = document.getElementById('searchInput').value.toLowerCase();
    const category = document.getElementById('filterCategory').value;
    
    const filtered = allRepos.filter(repo => {
        const readme = readmeCache[repo.name] || {};
        const searchText = (
            repo.name + ' ' + 
            repo.title + ' ' +
            repo.description + ' ' + 
            (readme.abstract || '') + ' ' + 
            (readme.highlights || []).join(' ') + ' ' +
            (readme.technologies || []).join(' ')
        ).toLowerCase();
        
        const matchesSearch = searchTerm === '' || searchText.includes(searchTerm);
        const matchesCategory = category === 'all' || repo.category === category;
        
        return matchesSearch && matchesCategory;
    });
    
    // Sort filtered results by stars (descending)
    filtered.sort((a, b) => b.stars - a.stars);
    
    displayRepos(filtered);
}

// Expand README in modal
async function expandReadme(repoName) {
    const readme = readmeCache[repoName];
    const repo = allRepos.find(r => r.name === repoName);
    
    if (!readme || !repo) {
        alert('README not available');
        return;
    }
    
    // Create modal with loading state initially
    const modal = document.createElement('div');
    modal.className = 'modal';
    modal.innerHTML = `
        <div class="modal-content">
            <span class="modal-close" onclick="this.parentElement.parentElement.remove()">&times;</span>
            <h2>${readme.title || repoName}</h2>
            <div class="modal-body">
                <div class="loading">
                    <div class="spinner"></div>
                    <p>Loading README content...</p>
                </div>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    
    // Try to fetch real README content
    let readmeContent = '';
    if (repo.readme_file) {
        // Build the absolute URL based on the current location
        let readmeUrl;
        
        if (window.location.hostname === 'hwai-collab.github.io') {
            // On GitHub Pages, use absolute URL
            readmeUrl = `https://hwai-collab.github.io/HWAI/${repo.readme_file}`;
        } else if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
            // Local development
            readmeUrl = `/${repo.readme_file}`;
        } else {
            // Fallback to relative path
            readmeUrl = repo.readme_file;
        }
        
        console.log('Fetching README preview from:', readmeUrl);
        
        try {
            const response = await fetch(readmeUrl);
            if (response.ok) {
                const fullContent = await response.text();
                // Extract first few sections for preview (up to 2000 characters)
                const lines = fullContent.split('\n');
                const previewLines = [];
                let charCount = 0;
                
                for (let line of lines) {
                    if (charCount + line.length > 2000) {
                        previewLines.push('...\n\n*[Content truncated - Download full README for complete documentation]*');
                        break;
                    }
                    previewLines.push(line);
                    charCount += line.length + 1;
                }
                
                readmeContent = previewLines.join('\n');
                console.log('Successfully loaded README preview');
            } else {
                console.error('Failed to fetch README - status:', response.status);
            }
        } catch (error) {
            console.error('Error fetching README:', error);
        }
    }
    
    // Update modal with content
    const modalBody = modal.querySelector('.modal-body');
    if (readmeContent) {
        // Convert markdown to HTML for basic display
        const htmlContent = readmeContent
            .replace(/^### (.*$)/gm, '<h4>$1</h4>')
            .replace(/^## (.*$)/gm, '<h3>$1</h3>')
            .replace(/^# (.*$)/gm, '<h2>$1</h2>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/^\- (.*$)/gm, '<li>$1</li>')
            .replace(/\n\n/g, '</p><p>')
            .replace(/^(?!<[h|l|p])/gm, '<p>')
            .replace(/(?<!>)$/gm, '</p>');
        
        modalBody.innerHTML = `
            <div class="readme-preview" style="max-height: 60vh; overflow-y: auto; line-height: 1.6;">
                ${htmlContent}
            </div>
            <p class="modal-note" style="margin-top: 2rem; padding: 1rem; background: var(--background); border-left: 4px solid var(--accent-color);">
                This is a preview of the actual README file. Download the complete README for full documentation.
            </p>
            <div class="modal-footer" style="margin-top: 2rem;">
                <button class="btn-primary" onclick="requestAccess('${repoName}')">Request Source Code Access</button>
                <button class="btn-secondary" onclick="downloadReadme('${repoName}')">Download Complete README</button>
            </div>
        `;
    } else {
        // Fallback to metadata display
        modalBody.innerHTML = `
            <div class="modal-abstract">
                <h3>Abstract</h3>
                <p>${readme.abstract}</p>
            </div>
            
            ${readme.highlights ? `
            <div class="modal-highlights">
                <h3>Key Features</h3>
                <ul>
                    ${readme.highlights.map(highlight => `<li>${highlight}</li>`).join('')}
                </ul>
            </div>
            ` : ''}
            
            ${readme.sections ? `
            <div class="modal-sections">
                <h3>Documentation Sections</h3>
                <ul>
                    ${readme.sections.map(section => `<li>${section}</li>`).join('')}
                </ul>
            </div>
            ` : ''}
            
            ${readme.technologies ? `
            <div class="modal-technologies">
                <h3>Technologies Used</h3>
                <div class="tech-list">
                    ${readme.technologies.map(tech => `<span class="tech-badge-large">${tech}</span>`).join('')}
                </div>
            </div>
            ` : ''}
            
            <p class="modal-note">To access the complete source code and documentation, please request access using the button below.</p>
            <div class="modal-footer">
                <button class="btn-primary" onclick="requestAccess('${repoName}')">Request Source Code Access</button>
                <button class="btn-secondary" onclick="downloadReadme('${repoName}')">Download README</button>
            </div>
        `;
    }
}

// Setup access request modal
function setupAccessRequestModal() {
    // Create modal HTML if it doesn't exist
    if (!document.getElementById('accessModal')) {
        const accessModal = document.createElement('div');
        accessModal.id = 'accessModal';
        accessModal.className = 'modal';
        accessModal.style.display = 'none';
        accessModal.innerHTML = `
            <div class="modal-content access-modal">
                <span class="modal-close" onclick="closeAccessModal()">&times;</span>
                <h2>Request Source Code Access</h2>
                <div class="modal-body">
                    <p>Please provide your details to request access to the source code for <strong id="requestedRepo"></strong>.</p>
                    
                    <form id="accessRequestForm" onsubmit="submitAccessRequest(event)">
                        <div class="form-group">
                            <label for="requesterName">Name *</label>
                            <input type="text" id="requesterName" name="name" required>
                        </div>
                        
                        <div class="form-group">
                            <label for="requesterEmail">Email *</label>
                            <input type="email" id="requesterEmail" name="email" required>
                        </div>
                        
                        <div class="form-group">
                            <label for="requesterOrganization">Organization</label>
                            <input type="text" id="requesterOrganization" name="organization">
                        </div>
                        
                        <div class="form-group">
                            <label for="requesterPurpose">Purpose of Access *</label>
                            <textarea id="requesterPurpose" name="purpose" rows="4" required 
                                placeholder="Please describe how you intend to use this source code..."></textarea>
                        </div>
                        
                        <div class="form-group">
                            <label for="requesterRole">Your Role</label>
                            <select id="requesterRole" name="role">
                                <option value="">Select...</option>
                                <option value="researcher">Researcher</option>
                                <option value="developer">Developer</option>
                                <option value="student">Student</option>
                                <option value="healthcare">Healthcare Professional</option>
                                <option value="government">Government Official</option>
                                <option value="nonprofit">Non-Profit Organization</option>
                                <option value="commercial">Commercial Entity</option>
                                <option value="other">Other</option>
                            </select>
                        </div>
                        
                        <div class="form-actions">
                            <button type="submit" class="btn-primary">Submit Request</button>
                            <button type="button" class="btn-secondary" onclick="closeAccessModal()">Cancel</button>
                        </div>
                    </form>
                </div>
            </div>
        `;
        document.body.appendChild(accessModal);
    }
}

// Request access to source code
function requestAccess(repoName) {
    const modal = document.getElementById('accessModal');
    const repoNameElement = document.getElementById('requestedRepo');
    const repo = allRepos.find(r => r.name === repoName);
    
    if (repo) {
        repoNameElement.textContent = repo.title || repo.name;
        repoNameElement.dataset.repoName = repoName;
        modal.style.display = 'flex';
        
        // Close any other open modals
        document.querySelectorAll('.modal').forEach(m => {
            if (m.id !== 'accessModal') {
                m.remove();
            }
        });
    }
}

// Close access modal
function closeAccessModal() {
    const modal = document.getElementById('accessModal');
    modal.style.display = 'none';
    
    // Reset form
    document.getElementById('accessRequestForm').reset();
}

// Submit access request
async function submitAccessRequest(event) {
    event.preventDefault();
    
    const form = document.getElementById('accessRequestForm');
    const formData = new FormData(form);
    const repoName = document.getElementById('requestedRepo').dataset.repoName;
    const repo = allRepos.find(r => r.name === repoName);
    
    // Prepare email data
    const requestData = {
        repository: repo.title || repo.name,
        name: formData.get('name'),
        email: formData.get('email'),
        organization: formData.get('organization') || 'Not specified',
        purpose: formData.get('purpose'),
        role: formData.get('role') || 'Not specified',
        timestamp: new Date().toISOString()
    };
    
    // Send email using EmailJS or similar service
    // For now, we'll use a mailto link as a fallback
    const subject = `Source Code Access Request: ${requestData.repository}`;
    const body = `
New source code access request received:

Repository: ${requestData.repository}
Name: ${requestData.name}
Email: ${requestData.email}
Organization: ${requestData.organization}
Role: ${requestData.role}

Purpose of Access:
${requestData.purpose}

Timestamp: ${requestData.timestamp}
    `.trim();
    
    // Create mailto link
    const mailtoLink = `mailto:info@helloworldai.com.au?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;
    
    // Try to send via API if available, otherwise use mailto
    try {
        // If you have a backend API endpoint, use it here
        // const response = await fetch('/api/access-request', {
        //     method: 'POST',
        //     headers: {'Content-Type': 'application/json'},
        //     body: JSON.stringify(requestData)
        // });
        
        // For now, open mailto link
        window.location.href = mailtoLink;
        
        // Show success message
        alert(`Thank you for your request. We will review it and contact you at ${requestData.email} within 2-3 business days.`);
        
        // Close modal and reset form
        closeAccessModal();
        
    } catch (error) {
        console.error('Error submitting request:', error);
        alert('There was an error submitting your request. Please try again or contact us directly at info@helloworldai.com.au');
    }
}

// Download README as markdown file
async function downloadReadme(repoName) {
    const repo = allRepos.find(r => r.name === repoName);
    
    if (!repo) {
        alert('Repository not found');
        return;
    }
    
    let markdownContent;
    
    // Try to fetch the real README file first
    if (repo.readme_file) {
        // Build the absolute URL based on the current location
        let readmeUrl;
        
        if (window.location.hostname === 'hwai-collab.github.io') {
            // On GitHub Pages, use absolute URL
            readmeUrl = `https://hwai-collab.github.io/HWAI/${repo.readme_file}`;
        } else if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
            // Local development
            readmeUrl = `/${repo.readme_file}`;
        } else {
            // Fallback to relative path
            readmeUrl = repo.readme_file;
        }
        
        console.log('Downloading README from:', readmeUrl);
        
        try {
            const response = await fetch(readmeUrl);
            if (response.ok) {
                markdownContent = await response.text();
                console.log('Successfully fetched real README for download');
            } else {
                console.error('Failed to fetch README - HTTP status:', response.status);
                console.error('URL was:', readmeUrl);
                markdownContent = generateFallbackReadme(repo);
            }
        } catch (error) {
            console.error('Error fetching README:', error);
            console.error('URL was:', readmeUrl);
            markdownContent = generateFallbackReadme(repo);
        }
    } else {
        // Fallback to generated content from metadata
        markdownContent = generateFallbackReadme(repo);
    }
    
    // Create blob and download
    const blob = new Blob([markdownContent], { type: 'text/markdown;charset=utf-8' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${repo.name}-README.md`;
    
    // Trigger download
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    // Clean up
    window.URL.revokeObjectURL(url);
}

// Generate fallback README content from metadata
function generateFallbackReadme(repo) {
    const readme = readmeCache[repo.name];
    if (!readme) {
        return `# ${repo.name}\n\n${repo.description}\n\nREADME content not available.`;
    }
    
    let markdownContent = `# ${readme.title || repo.name}\n\n`;
    markdownContent += `**Category:** ${repo.category}\n`;
    markdownContent += `**Language:** ${repo.language || 'Multiple'}\n`;
    markdownContent += `**Stars:** ${repo.stars} | **Forks:** ${repo.forks}\n\n`;
    markdownContent += `## Abstract\n\n${readme.abstract || repo.description}\n\n`;
    
    if (readme.technologies && readme.technologies.length > 0) {
        markdownContent += `## Technologies\n\n`;
        readme.technologies.forEach(tech => {
            markdownContent += `- ${tech}\n`;
        });
        markdownContent += '\n';
    }
    
    if (readme.highlights && readme.highlights.length > 0) {
        markdownContent += `## Key Features\n\n`;
        readme.highlights.forEach(highlight => {
            markdownContent += `- ${highlight}\n`;
        });
        markdownContent += '\n';
    }
    
    return markdownContent;
}

// Download all READMEs as a single zip file (requires JSZip library)
function downloadAllReadmes() {
    // Check if JSZip is available
    if (typeof JSZip === 'undefined') {
        // Fall back to downloading a combined markdown file
        downloadCombinedReadme();
        return;
    }
    
    const zip = new JSZip();
    
    allRepos.forEach(repo => {
        const readme = readmeCache[repo.name];
        if (readme) {
            let content = generateReadmeContent(repo, readme);
            zip.file(`${repo.name}-README.md`, content);
        }
    });
    
    zip.generateAsync({ type: 'blob' }).then(function(content) {
        const url = window.URL.createObjectURL(content);
        const link = document.createElement('a');
        link.href = url;
        link.download = 'HWAI-Collaborative-READMEs.zip';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);
    });
}

// Download combined README file
function downloadCombinedReadme() {
    let combinedContent = `# HWAI Collaborative - Academic Research Portfolio\n\n`;
    combinedContent += `Healthcare, Wellbeing and AI Research Collaborative\n\n`;
    combinedContent += `**Total Projects:** ${allRepos.length}\n`;
    combinedContent += `**Downloaded:** ${new Date().toLocaleDateString()}\n\n`;
    combinedContent += `---\n\n`;
    combinedContent += `## Table of Contents\n\n`;
    
    // Add table of contents
    allRepos.forEach((repo, index) => {
        const readme = readmeCache[repo.name];
        combinedContent += `${index + 1}. [${readme.title || repo.name}](#${repo.name.toLowerCase().replace(/[^a-z0-9]/g, '-')})\n`;
    });
    
    combinedContent += `\n---\n\n`;
    
    // Add each project
    allRepos.forEach(repo => {
        if (repo.readme_content) {
            // Use actual README content
            combinedContent += repo.readme_content + '\n\n';
            combinedContent += `---\n\n`;
        } else {
            // Fallback to summary format
            const readme = readmeCache[repo.name];
            if (readme) {
                combinedContent += `## ${readme.title || repo.name}\n\n`;
                combinedContent += `**Category:** ${repo.category} | `;
                combinedContent += `**Language:** ${repo.language || 'Multiple'} | `;
                combinedContent += `**Stars:** ${repo.stars} | **Forks:** ${repo.forks}\n\n`;
                
                combinedContent += `### Abstract\n${readme.abstract || repo.description}\n\n`;
                
                if (readme.technologies && readme.technologies.length > 0) {
                    combinedContent += `### Technologies\n`;
                    combinedContent += readme.technologies.join(', ') + '\n\n';
                }
                
                if (readme.highlights && readme.highlights.length > 0) {
                    combinedContent += `### Key Features\n`;
                    readme.highlights.forEach(highlight => {
                        combinedContent += `- ${highlight}\n`;
                    });
                    combinedContent += '\n';
                }
                
                combinedContent += `---\n\n`;
            }
        }
    });
    
    // Add footer
    combinedContent += `## Contact Information\n\n`;
    combinedContent += `For source code access and collaboration inquiries:\n\n`;
    combinedContent += `- **Email:** info@helloworldai.com.au\n`;
    combinedContent += `- **Organization:** HWAI Collaborative\n`;
    combinedContent += `- **Website:** https://hwai-collab.github.io/HWAI/\n`;
    
    // Create and download
    const blob = new Blob([combinedContent], { type: 'text/markdown;charset=utf-8' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'HWAI-Collaborative-Complete-Portfolio.md';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
}

// Helper function to generate README content
function generateReadmeContent(repo, readme) {
    let content = `# ${readme.title || repo.name}\n\n`;
    content += `**Category:** ${repo.category}\n`;
    content += `**Language:** ${repo.language || 'Multiple'}\n`;
    content += `**Stars:** ${repo.stars} | **Forks:** ${repo.forks}\n\n`;
    content += `## Abstract\n\n${readme.abstract || repo.description}\n\n`;
    
    if (readme.technologies && readme.technologies.length > 0) {
        content += `## Technologies\n\n`;
        readme.technologies.forEach(tech => {
            content += `- ${tech}\n`;
        });
        content += '\n';
    }
    
    if (readme.highlights && readme.highlights.length > 0) {
        content += `## Key Features\n\n`;
        readme.highlights.forEach(highlight => {
            content += `- ${highlight}\n`;
        });
        content += '\n';
    }
    
    return content;
}

// View README popup with formatted content and download option
async function viewReadmePopup(repoName) {
    console.log('viewReadmePopup called with:', repoName);
    console.log('allRepos length:', allRepos.length);
    
    const repo = allRepos.find(r => r && r.name === repoName);
    if (!repo) {
        console.error('Repository not found:', repoName);
        alert('Repository not found');
        return;
    }
    
    console.log('Found repo:', repo);
    console.log('repo.readme_file:', repo.readme_file);
    
    // Simple filename construction
    let readmeFileName = `${repoName}.md`;
    
    console.log('Using filename:', readmeFileName);
    
    const readmeUrl = `https://hwai-collab.github.io/HWAI/ReadMe/${readmeFileName}`;
    
    try {
        const response = await fetch(readmeUrl);
        if (!response.ok) throw new Error('README not found');
        
        const readmeContent = await response.text();
        
        // Create modal
        const modal = document.createElement('div');
        modal.className = 'modal';
        modal.innerHTML = `
            <div class="modal-content readme-modal">
                <span class="modal-close">&times;</span>
                <div class="readme-header">
                    <h2>${repo.title || repo.name}</h2>
                    <button class="btn-primary" onclick="downloadReadmeFile('${repo.name}')">
                        <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                            <path d="M7.47 10.78a.75.75 0 001.06 0l3.75-3.75a.75.75 0 00-1.06-1.06L8.75 8.44V1.75a.75.75 0 00-1.5 0v6.69L4.78 5.97a.75.75 0 00-1.06 1.06l3.75 3.75zM3.75 13a.75.75 0 000 1.5h8.5a.75.75 0 000-1.5h-8.5z"/>
                        </svg>
                        Download README
                    </button>
                </div>
                <div class="readme-content">
                    <pre>${readmeContent}</pre>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        
        // Close modal functionality
        modal.querySelector('.modal-close').onclick = () => {
            document.body.removeChild(modal);
        };
        
        modal.onclick = (e) => {
            if (e.target === modal) {
                document.body.removeChild(modal);
            }
        };
        
    } catch (error) {
        alert('Unable to load README content. Please try again.');
    }
}

// Download README file (original function renamed)
async function downloadReadmeFile(repoName) {
    const repo = allRepos.find(r => r.name === repoName);
    if (!repo) {
        console.error('Repository not found for download:', repoName);
        return;
    }
    
    // Handle missing readme_file property
    let readmeFileName;
    if (repo.readme_file && typeof repo.readme_file === 'string') {
        readmeFileName = repo.readme_file.replace('ReadMe/', '');
    } else {
        readmeFileName = `${repo.name}.md`;
    }
    
    const readmeUrl = `https://hwai-collab.github.io/HWAI/ReadMe/${readmeFileName}`;
    
    try {
        const response = await fetch(readmeUrl);
        if (!response.ok) throw new Error('README not found');
        
        const content = await response.text();
        const blob = new Blob([content], { type: 'text/markdown' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `${repo.name}-README.md`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    } catch (error) {
        alert('Unable to download README. Please try again.');
    }
}