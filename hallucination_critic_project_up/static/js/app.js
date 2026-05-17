/**
 * Hallucination Critic - Frontend JavaScript
 * Handles user interactions and API communication
 */

// Main analyze function
function analyzeSummary() {
    const sourceText = document.getElementById('sourceText').value.trim();
    const summaryText = document.getElementById('summaryText').value.trim();

    if (!sourceText || !summaryText) {
        alert('⚠️ Please enter both source text and summary text.');
        return;
    }

    // Show loading overlay
    document.getElementById('loadingOverlay').classList.add('active');

    // Send request to backend
    fetch('/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            source_text: sourceText,
            summary_text: summaryText
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert('Error: ' + data.error);
            document.getElementById('loadingOverlay').classList.remove('active');
            return;
        }

        displayResults(data);
        
        document.getElementById('loadingOverlay').classList.remove('active');
        document.getElementById('resultsContainer').classList.add('active');
        
        // Scroll to results
        document.getElementById('resultsContainer').scrollIntoView({ behavior: 'smooth' });
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Failed to analyze. Please try again.');
        document.getElementById('loadingOverlay').classList.remove('active');
    });
}

// Display results from backend
function displayResults(analysis) {
    const isClean = analysis.verdict === 'SUMMARY VERIFIED';
    
    // Update verdict card
    const verdictCard = document.getElementById('verdictCard');
    verdictCard.className = 'verdict-card ' + (isClean ? 'safe' : 'danger');
    
    document.getElementById('verdictIcon').textContent = isClean ? '✅' : '⚠️';
    document.getElementById('verdictTitle').textContent = analysis.verdict;
    document.getElementById('verdictSubtitle').textContent = isClean 
        ? 'All facts verified against source text' 
        : 'Fabricated information detected in summary';
    
    // Update metrics
    const hallucinationScore = analysis.hallucination_score * 100;
    const semanticSimilarity = analysis.semantic_similarity * 100;
    const keywordCoverage = analysis.keyword_analysis.coverage_rate * 100;
    
    updateMetric('hallucinationScore', 'hallucinationBar', hallucinationScore);
    updateMetric('semanticSim', 'semanticBar', semanticSimilarity);
    updateMetric('keywordCoverage', 'keywordBar', keywordCoverage);
    
    // Generate analysis cards
    const analysisGrid = document.getElementById('analysisGrid');
    analysisGrid.innerHTML = '';
    
    // Entity Analysis
    const entityAnalysis = analysis.entity_analysis;
    const hasEntities = Object.keys(entityAnalysis.hallucinated).length > 0 || 
                       Object.keys(entityAnalysis.verified).length > 0;
    
    if (hasEntities) {
        analysisGrid.innerHTML += createAnalysisCard(
            '🏷️', 
            'Entity Analysis',
            generateEntityFindings(entityAnalysis)
        );
    }
    
    // Numerical Analysis
    const numericalAnalysis = analysis.numerical_analysis;
    const hasNumbers = numericalAnalysis.hallucinated.length > 0 || 
                      numericalAnalysis.verified.length > 0;
    
    if (hasNumbers) {
        analysisGrid.innerHTML += createAnalysisCard(
            '🔢',
            'Numerical Verification',
            generateNumberFindings(numericalAnalysis)
        );
    }
    
    // Overall Assessment (only shown when hallucination detected)
    if (!isClean) {
        analysisGrid.innerHTML += createAnalysisCard(
            '📊',
            'Overall Assessment',
            generateOverallAssessment(analysis)
        );
    }

    // Computational Trace
    if (analysis.computational_trace) {
        analysisGrid.innerHTML += generateComputationalTrace(analysis.computational_trace);
    }

    // False reasons appear after the trace (only when hallucination detected)
    if (!isClean && analysis.false_reasons && analysis.false_reasons.length > 0) {
        analysisGrid.innerHTML += generateFalseReasonsCard(analysis.false_reasons);
    }
}

function updateMetric(valueId, barId, value) {
    document.getElementById(valueId).textContent = value.toFixed(1) + '%';
    setTimeout(() => {
        document.getElementById(barId).style.width = value + '%';
    }, 100);
}

function createAnalysisCard(icon, title, content) {
    return `
        <div class="analysis-card">
            <div class="analysis-header">
                <div class="analysis-icon">${icon}</div>
                <div class="analysis-title">${title}</div>
            </div>
            ${content}
        </div>
    `;
}

function generateEntityFindings(entityAnalysis) {
    let html = '';
    
    // Flatten hallucinated entities
    const hallucinatedEntities = [];
    for (const [type, entities] of Object.entries(entityAnalysis.hallucinated)) {
        hallucinatedEntities.push(...entities);
    }
    
    // Flatten verified entities
    const verifiedEntities = [];
    for (const [type, entities] of Object.entries(entityAnalysis.verified)) {
        verifiedEntities.push(...entities);
    }
    
    if (hallucinatedEntities.length > 0) {
        html += `
            <div class="finding-item error">
                <div class="finding-label">⚠️ Fabricated Entities</div>
                <div class="finding-content">${hallucinatedEntities.join(', ')}</div>
                <div class="finding-reason">These entities do not appear in the source text</div>
            </div>
        `;
    }
    
    if (verifiedEntities.length > 0) {
        html += `
            <div class="finding-item success">
                <div class="finding-label">✅ Verified Entities</div>
                <div class="finding-content">${verifiedEntities.join(', ')}</div>
                <div class="finding-reason">All these entities found in source</div>
            </div>
        `;
    }
    
    if (!html) {
        html = '<div class="finding-content" style="color: #9ca3af;">No entities detected in summary.</div>';
    }
    
    return html;
}

function generateNumberFindings(numericalAnalysis) {
    let html = '';
    
    if (numericalAnalysis.hallucinated.length > 0) {
        html += `
            <div class="finding-item error">
                <div class="finding-label">⚠️ Fabricated Numbers</div>
                <div class="finding-content">${numericalAnalysis.hallucinated.join(', ')}</div>
                <div class="finding-reason">Source contains: ${numericalAnalysis.source_numbers.join(', ')}</div>
            </div>
        `;
    }
    
    if (numericalAnalysis.verified.length > 0) {
        html += `
            <div class="finding-item success">
                <div class="finding-label">✅ Verified Numbers</div>
                <div class="finding-content">${numericalAnalysis.verified.join(', ')}</div>
                <div class="finding-reason">All numbers match source text</div>
            </div>
        `;
    }
    
    if (!html) {
        html = '<div class="finding-content" style="color: #9ca3af;">No numbers detected in summary.</div>';
    }
    
    return html;
}

function generateOverallAssessment(analysis) {
    const isClean = analysis.verdict === 'SUMMARY VERIFIED';
    
    if (isClean) {
        return `
            <div class="finding-item success">
                <div class="finding-label">✅ Quality Assessment</div>
                <div class="finding-content">
                    The summary appears to be factually grounded in the source text. 
                    All key entities and numbers are verified. The summary maintains 
                    accuracy while condensing the information.
                </div>
            </div>
        `;
    } else {
        return `
            <div class="finding-item error">
                <div class="finding-label">⚠️ Hallucination Warning</div>
                <div class="finding-content">
                    The summary contains fabricated information not present in the source text. 
                    This could mislead readers and should be corrected before publication. 
                    Review the flagged entities and numbers above.
                </div>
            </div>
        `;
    }
}

function generateFalseReasonsCard(reasons) {
    const items = reasons.map((r, i) => `
        <div class="false-reason-item">
            <div class="false-reason-num">${i + 1}</div>
            <div>${escapeHtml(r)}</div>
        </div>
    `).join('');

    return `
        <div class="false-reasons-card">
            <div class="false-reasons-title">
                <span>⛔</span> Why This Summary Is Flagged as False
            </div>
            ${items}
        </div>
    `;
}

function generateComputationalTrace(trace) {
    const stepsHtml = trace.steps.map(s => {
        const statusClass = s.status;
        const badge = s.status === 'pass' ? '✓ PASS' : s.status === 'fail' ? '✗ FAIL' : '— SKIP';

        let extras = '';
        if (s.hallucinated && s.hallucinated.length > 0) {
            extras += `<div class="trace-step-detail">Hallucinated: <span style="color:#f87171">${escapeHtml(s.hallucinated.join(', '))}</span></div>`;
        }
        if (s.verified && s.verified.length > 0) {
            extras += `<div class="trace-step-detail">Verified: <span style="color:#34d399">${escapeHtml(s.verified.join(', '))}</span></div>`;
        }
        if (s.source_numbers && s.source_numbers.length > 0) {
            extras += `<div class="trace-step-detail">Source numbers: ${escapeHtml(s.source_numbers.join(', '))}</div>`;
        }
        if (s.covered && s.covered.length > 0) {
            extras += `<div class="trace-step-detail">Covered keywords: <span style="color:#34d399">${escapeHtml(s.covered.join(', '))}</span></div>`;
        }
        if (s.missing && s.missing.length > 0) {
            extras += `<div class="trace-step-detail">Missing keywords: <span style="color:#f87171">${escapeHtml(s.missing.join(', '))}</span></div>`;
        }

        const contribLine = s.contribution !== null && s.contribution !== undefined
            ? `<div class="trace-contribution">Contribution to score: <span>${s.contribution.toFixed(4)}</span></div>`
            : '';

        return `
            <div class="trace-step ${statusClass}">
                <div class="trace-step-title">Step ${s.step}: ${escapeHtml(s.name)} &nbsp; <span style="font-size:0.8em;opacity:0.7">${badge}</span></div>
                <div class="trace-step-detail">${escapeHtml(s.detail)}</div>
                ${extras}
                <div class="trace-formula">${escapeHtml(s.formula)}</div>
                ${contribLine}
            </div>
        `;
    }).join('');

    const agg = trace.aggregation;
    const verdictClass = agg.verdict === 'SUMMARY VERIFIED' ? 'safe' : 'danger';

    return `
        <div class="trace-card">
            <div class="trace-header">
                <span style="font-size:1.4em">🧮</span>
                <div class="trace-title">Computational Trace</div>
            </div>
            ${stepsHtml}
            <div class="trace-aggregation">
                <div class="trace-aggregation-title">Final Score Aggregation</div>
                <div class="trace-aggregation-formula">${escapeHtml(agg.formula)}</div>
                <div class="trace-threshold">Detection threshold: ${(agg.threshold * 100).toFixed(0)}% &nbsp;|&nbsp; Final score: ${(agg.final_score * 100).toFixed(1)}%</div>
                <div class="trace-verdict-line ${verdictClass}">${escapeHtml(agg.verdict)}</div>
            </div>
        </div>
    `;
}

function escapeHtml(str) {
    if (typeof str !== 'string') return String(str);
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// Load example when page loads (optional)
document.addEventListener('DOMContentLoaded', function() {
    console.log('Hallucination Critic loaded successfully');
});