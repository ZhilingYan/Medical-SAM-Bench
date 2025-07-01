// Copy to clipboard functionality
function copyToClipboard() {
    const codeElement = document.querySelector('.citation-box code');
    const textToCopy = codeElement.textContent;
    
    // Create a temporary textarea element
    const textarea = document.createElement('textarea');
    textarea.value = textToCopy;
    textarea.style.position = 'absolute';
    textarea.style.left = '-9999px';
    document.body.appendChild(textarea);
    
    // Select and copy the text
    textarea.select();
    document.execCommand('copy');
    
    // Remove the temporary element
    document.body.removeChild(textarea);
    
    // Update button text
    const copyButton = document.querySelector('.copy-button');
    const originalHTML = copyButton.innerHTML;
    copyButton.innerHTML = '<i class="fas fa-check"></i> Copied!';
    copyButton.style.backgroundColor = '#10b981';
    
    // Reset button after 2 seconds
    setTimeout(() => {
        copyButton.innerHTML = originalHTML;
        copyButton.style.backgroundColor = '';
    }, 2000);
}

// Smooth scrolling for anchor links
document.addEventListener('DOMContentLoaded', function() {
    // Smooth scroll for navigation links
    const links = document.querySelectorAll('a[href^="#"]');
    links.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            const targetElement = document.getElementById(targetId);
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Add scroll animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -100px 0px'
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-in');
            }
        });
    }, observerOptions);
    
    // Observe all sections and contributions
    const elementsToAnimate = document.querySelectorAll('section, .contribution, .stat');
    elementsToAnimate.forEach(el => {
        observer.observe(el);
    });
    
    // Add hover effects to images
    const images = document.querySelectorAll('.hero-image, .method-image, .results-image, .demo-image');
    images.forEach(img => {
        img.addEventListener('click', function() {
            // Create modal for image viewing
            const modal = document.createElement('div');
            modal.className = 'image-modal';
            modal.innerHTML = `
                <div class="modal-content">
                    <span class="close-modal">&times;</span>
                    <img src="${this.src}" alt="${this.alt}">
                </div>
            `;
            document.body.appendChild(modal);
            
            // Close modal functionality
            modal.addEventListener('click', function(e) {
                if (e.target === modal || e.target.className === 'close-modal') {
                    modal.remove();
                }
            });
        });
    });
    
    // Add loading animation
    window.addEventListener('load', function() {
        document.body.classList.add('loaded');
    });
});

// Add CSS for animations dynamically
const style = document.createElement('style');
style.textContent = `
    /* Loading animation */
    body {
        opacity: 0;
        transition: opacity 0.5s ease;
    }
    
    body.loaded {
        opacity: 1;
    }
    
    /* Scroll animations */
    section, .contribution, .stat {
        opacity: 0;
        transform: translateY(30px);
        transition: opacity 0.6s ease, transform 0.6s ease;
    }
    
    section.animate-in, .contribution.animate-in, .stat.animate-in {
        opacity: 1;
        transform: translateY(0);
    }
    
    /* Image modal */
    .image-modal {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.9);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 1000;
        animation: fadeIn 0.3s ease;
    }
    
    .modal-content {
        position: relative;
        max-width: 90%;
        max-height: 90%;
    }
    
    .modal-content img {
        width: 100%;
        height: auto;
        border-radius: 8px;
    }
    
    .close-modal {
        position: absolute;
        top: -40px;
        right: 0;
        color: white;
        font-size: 2rem;
        cursor: pointer;
        transition: color 0.3s ease;
    }
    
    .close-modal:hover {
        color: #f59e0b;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    /* Hover effect for clickable images */
    .hero-image, .method-image, .results-image, .demo-image {
        cursor: pointer;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .hero-image:hover, .method-image:hover, .results-image:hover, .demo-image:hover {
        transform: scale(1.02);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    }
`;
document.head.appendChild(style); 