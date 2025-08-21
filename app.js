// HWAI Academic Portfolio - Repository Fetcher
const GITHUB_ORG = 'HWAI-collab';
const GITHUB_API_BASE = 'https://api.github.com';

// Cache for repositories and READMEs
let allRepos = [];
let readmeCache = {};

// Initialize the application
document.addEventListener('DOMContentLoaded', init);

async function init() {
    await loadRepositories();
    setupEventListeners();
}

// Setup event listeners for search and filter
function setupEventListeners() {
    const searchInput = document.getElementById('searchInput');
    const filterCategory = document.getElementById('filterCategory');
    
    searchInput.addEventListener('input', filterRepos);
    filterCategory.addEventListener('change', filterRepos);
}

// Load all repositories from HWAI-collab
async function loadRepositories() {
    const loading = document.getElementById('loading');
    const errorMessage = document.getElementById('errorMessage');
    const repoGrid = document.getElementById('repoGrid');
    
    try {
        // Fetch organization repositories
        const response = await fetch(`${GITHUB_API_BASE}/orgs/${GITHUB_ORG}/repos?per_page=100&sort=updated`);
        
        if (!response.ok) {
            throw new Error('Failed to fetch repositories');
        }
        
        const repos = await response.json();
        
        // Process each repository
        for (const repo of repos) {
            const repoData = {
                name: repo.name,
                description: repo.description || 'No description available',
                url: repo.html_url,
                stars: repo.stargazers_count,
                forks: repo.forks_count,
                language: repo.language,
                updated: new Date(repo.updated_at),
                topics: repo.topics || [],
                category: categorizeRepo(repo)
            };
            
            allRepos.push(repoData);
            
            // Fetch README content
            await fetchReadme(repoData);
        }
        
        // Display repositories
        displayRepos(allRepos);
        loading.style.display = 'none';
        
    } catch (error) {
        console.error('Error loading repositories:', error);
        loading.style.display = 'none';
        errorMessage.style.display = 'block';
    }
}

// Categorize repository based on name and topics
function categorizeRepo(repo) {
    const name = repo.name.toLowerCase();
    const topics = repo.topics || [];
    const allText = name + ' ' + topics.join(' ');
    
    if (allText.includes('ai') || allText.includes('ml') || allText.includes('neural')) {
        return 'ai';
    } else if (allText.includes('health') || allText.includes('medical') || allText.includes('care')) {
        return 'healthcare';
    } else if (allText.includes('youth') || allText.includes('slnq') || allText.includes('young')) {
        return 'youth';
    } else if (allText.includes('infra') || allText.includes('platform') || allText.includes('system')) {
        return 'infrastructure';
    }
    return 'other';
}

// Fetch README content for a repository
async function fetchReadme(repo) {
    try {
        // Try multiple README formats
        const readmeFormats = ['README.md', 'readme.md', 'README.MD'];
        
        for (const format of readmeFormats) {
            const response = await fetch(`https://raw.githubusercontent.com/${GITHUB_ORG}/${repo.name}/main/${format}`);
            
            if (response.ok) {
                const content = await response.text();
                readmeCache[repo.name] = parseReadme(content);
                return;
            }
        }
        
        // If no README found in main, try master branch
        for (const format of readmeFormats) {
            const response = await fetch(`https://raw.githubusercontent.com/${GITHUB_ORG}/${repo.name}/master/${format}`);
            
            if (response.ok) {
                const content = await response.text();
                readmeCache[repo.name] = parseReadme(content);
                return;
            }
        }
        
        readmeCache[repo.name] = {
            title: repo.name,
            abstract: repo.description || 'No README available',
            sections: []
        };
        
    } catch (error) {
        console.error(`Error fetching README for ${repo.name}:`, error);
        readmeCache[repo.name] = {
            title: repo.name,
            abstract: repo.description || 'Error loading README',
            sections: []
        };
    }
}

// Parse README content to extract key information
function parseReadme(content) {
    const lines = content.split('\\n');
    const result = {
        title: '',
        abstract: '',
        sections: []
    };
    
    // Extract title (first # heading)
    const titleMatch = content.match(/^#\\s+(.+)$/m);
    if (titleMatch) {
        result.title = titleMatch[1];
    }
    
    // Extract abstract (first paragraph after title)
    const abstractMatch = content.match(/^#[^\\n]+\\n\\n([^#\\n]+)/);
    if (abstractMatch) {
        result.abstract = abstractMatch[1].trim();
    } else {
        // Fallback: get first non-empty paragraph
        const paragraphs = content.split('\\n\\n').filter(p => p.trim() && !p.startsWith('#'));
        if (paragraphs.length > 0) {
            result.abstract = paragraphs[0].substring(0, 300) + '...';
        }
    }
    
    // Extract main sections
    const sectionMatches = content.matchAll(/^##\\s+(.+)$/gm);
    for (const match of sectionMatches) {
        result.sections.push(match[1]);
    }
    
    return result;
}

// Display repositories in the grid
function displayRepos(repos) {
    const repoGrid = document.getElementById('repoGrid');
    repoGrid.innerHTML = '';
    
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
        
        ${readme.sections && readme.sections.length > 0 ? `
        <div class="card-sections">
            <h3>Key Sections</h3>
            <ul>
                ${readme.sections.slice(0, 5).map(section => `<li>${section}</li>`).join('')}
            </ul>
        </div>
        ` : ''}
        
        <div class="card-meta">
            <span class="meta-item">
                <svg class="icon" viewBox="0 0 16 16" width="16" height="16">
                    <path d="M8 .25a.75.75 0 01.673.418l1.882 3.815 4.21.612a.75.75 0 01.416 1.279l-3.046 2.97.719 4.192a.75.75 0 01-1.088.791L8 12.347l-3.766 1.98a.75.75 0 01-1.088-.79l.72-4.194L.818 6.374a.75.75 0 01.416-1.28l4.21-.611L7.327.668A.75.75 0 018 .25z"/>
                </svg>
                ${repo.stars}
            </span>
            <span class="meta-item">
                <svg class="icon" viewBox="0 0 16 16" width="16" height="16">
                    <path d="M5 3.25a.75.75 0 11-1.5 0 .75.75 0 011.5 0zm0 2.122a2.25 2.25 0 10-1.5 0v.878A2.25 2.25 0 005.75 8.5h1.5v2.128a2.251 2.251 0 101.5 0V8.5h1.5a2.25 2.25 0 002.25-2.25v-.878a2.25 2.25 0 10-1.5 0v.878a.75.75 0 01-.75.75h-4.5A.75.75 0 015 6.25v-.878zm3.75 7.378a.75.75 0 11-1.5 0 .75.75 0 011.5 0zm3-8.75a.75.75 0 100-1.5.75.75 0 000 1.5z"/>
                </svg>
                ${repo.forks}
            </span>
            ${repo.language ? `<span class="meta-item language">${repo.language}</span>` : ''}
        </div>
        
        <div class="card-footer">
            <a href="${repo.url}" target="_blank" class="btn-primary">View Repository</a>
            <button class="btn-secondary" onclick="expandReadme('${repo.name}')">Full README</button>
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
            repo.description + ' ' + 
            (readme.title || '') + ' ' + 
            (readme.abstract || '') + ' ' +
            (readme.sections || []).join(' ')
        ).toLowerCase();
        
        const matchesSearch = searchTerm === '' || searchText.includes(searchTerm);
        const matchesCategory = category === 'all' || repo.category === category;
        
        return matchesSearch && matchesCategory;
    });
    
    displayRepos(filtered);
}

// Expand README in modal (for future enhancement)
function expandReadme(repoName) {
    const readme = readmeCache[repoName];
    if (!readme) {
        alert('README not available');
        return;
    }
    
    // Create modal with full README content
    const modal = document.createElement('div');
    modal.className = 'modal';
    modal.innerHTML = `
        <div class="modal-content">
            <span class="modal-close" onclick="this.parentElement.parentElement.remove()">&times;</span>
            <h2>${readme.title || repoName}</h2>
            <div class="modal-body">
                <p>${readme.abstract}</p>
                ${readme.sections ? `
                <h3>Contents</h3>
                <ul>
                    ${readme.sections.map(section => `<li>${section}</li>`).join('')}
                </ul>
                ` : ''}
                <p class="modal-note">Visit the repository for the complete documentation.</p>
            </div>
            <div class="modal-footer">
                <a href="https://github.com/${GITHUB_ORG}/${repoName}" target="_blank" class="btn-primary">Open Repository</a>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
}