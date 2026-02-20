/**
 * Implements the client-side logic for a search ad learning system. 
 * It manages the taxonomy of categories and tags, handles user interactions,
 * communicates with the backend for predictions and feedback, and renders 
 * visualizations such as the tag cloud and category segments.
 */

const taxonomy = {
  "/Arts & Entertainment": {
    tags: ["indie film", "festival tickets", "stand-up comedy", "live concert", "streaming series", "retro arcade", "celebrity news", "movie trailers", "music production", "fan conventions"],
    trainingQueries: ["best comedy movies", "new music releases", "gaming stream highlights", "concerts near me", "funny podcast clips", "film festival schedule", "movie soundtrack vinyl", "stand up tour dates"]
  },
  "/Autos & Vehicles": {
    tags: ["electric SUV", "truck accessories", "motorcycle helmets", "fleet leasing", "cargo vans", "off-road tires", "car detailing", "vehicle financing", "used pickup", "diesel maintenance"],
    trainingQueries: ["best suv for families", "truck towing capacity", "motorcycle gear shop", "commercial van lease", "compare hybrid cars", "off road wheel kits", "auto detailing service", "fleet vehicle insurance"]
  },
  "/Beauty & Fitness": {
    tags: ["hydrating serum", "hair growth kit", "organic makeup", "spa packages", "pilates studio", "yoga retreat", "protein isolate", "bodybuilding split", "skin clinic", "lash extensions"],
    trainingQueries: ["best skincare routine", "hair care products", "yoga classes nearby", "bodybuilding workout plan", "spa day deals", "clean beauty brands", "protein powder review", "pilates reformer classes"]
  },
  "/Books & Literature": {
    tags: ["ebook subscriptions", "poetry anthology", "children's classics", "mystery novels", "book club picks", "indie magazines", "author interviews", "graphic novels", "audiobook bundles", "short story collection"],
    trainingQueries: ["best mystery novels", "ebooks for kids", "poetry books to read", "literary magazine subscription", "book club recommendations", "audiobook app", "new fantasy author", "children literature list"]
  },
  "/Business & Industrial": {
    tags: ["b2b lead gen", "trade show booth", "supply chain tools", "manufacturing ERP", "industrial textiles", "warehouse automation", "corporate event planner", "market research", "logistics platform", "branding agency"],
    trainingQueries: ["advertising agency services", "corporate event venue", "logistics software", "manufacturing consulting", "industrial textile suppliers", "warehouse management tools", "b2b marketing strategy", "trade show design"]
  },
  "/Computers & Electronics": {
    tags: ["custom pc build", "mesh wifi", "gaming monitor", "noise cancelling earbuds", "dev laptop", "network security", "ssd upgrade", "smart home hub", "graphics card", "usb-c docking"],
    trainingQueries: ["best laptop for coding", "pc hardware deals", "wifi router comparison", "consumer electronics sale", "network troubleshooting", "smart home setup", "ssd for gaming", "new graphics card"]
  },
  "/Finance": {
    tags: ["high-yield savings", "student loan refinance", "retirement calculator", "crypto portfolio", "credit monitoring", "small business banking", "term life insurance", "mortgage rates", "index fund strategy", "tax optimization"],
    trainingQueries: ["best investing apps", "bank account bonus", "insurance policy comparison", "loan interest rates", "how to build credit", "mortgage payment calculator", "retirement planning", "tax planning tips"]
  },
  "/Food & Drink": {
    tags: ["meal prep ideas", "specialty coffee", "vegan recipes", "fine dining", "mocktail recipes", "air fryer meals", "craft beverages", "food delivery", "kitchen gadgets", "dessert baking"],
    trainingQueries: ["easy dinner recipes", "best coffee beans", "restaurants near me", "healthy meal prep", "cocktail recipes", "baking ideas", "food delivery discounts", "kitchen tools review"]
  },
  "/Games": {
    tags: ["battle royale", "mmorpg builds", "mobile gacha", "casino bonus", "esports streams", "speedrun guides", "coop shooters", "gaming skins", "indie platformer", "tournament brackets"],
    trainingQueries: ["best online games", "video game tournaments", "casino app bonuses", "multiplayer shooter tips", "new indie games", "esports schedule", "game pass deals", "ranked ladder strategy"]
  },
  "/Health": {
    tags: ["telehealth visits", "vitamin plans", "mental wellness", "heart health", "nutrition coaching", "sleep tracking", "injury rehab", "preventive care", "health screening", "meditation apps"],
    trainingQueries: ["medical clinic near me", "wellness supplements", "healthy nutrition plan", "mental health resources", "fitness and wellness", "sleep improvement tips", "preventive checkups", "rehab exercises"]
  },
  "/Hobbies & Leisure": {
    tags: ["camping gear", "trail maps", "drone photography", "collectible antiques", "woodworking tools", "fishing reels", "birdwatching lenses", "board game nights", "craft supplies", "pottery class"],
    trainingQueries: ["camping checklist", "outdoor hobbies", "photography gear", "antique market", "woodworking projects", "fishing equipment", "pottery workshops", "board game groups"]
  },
  "/Home & Garden": {
    tags: ["indoor plants", "kitchen remodel", "diy shelving", "smart thermostat", "home decor", "lawn care", "patio furniture", "paint palettes", "garden irrigation", "tile backsplash"],
    trainingQueries: ["home improvement ideas", "garden planning guide", "interior design styles", "diy home projects", "best lawn tools", "kitchen renovation cost", "patio decor", "smart home upgrades"]
  },
  "/Internet & Telecom": {
    tags: ["seo audits", "email campaigns", "social analytics", "domain hosting", "5g plans", "web design agency", "landing page optimization", "content calendar", "creator tools", "broadband deals"],
    trainingQueries: ["web design services", "seo keyword tools", "email marketing platform", "social media strategy", "best mobile plans", "internet provider comparison", "content marketing", "domain registration"]
  },
  "/Jobs & Education": {
    tags: ["resume templates", "interview prep", "online certifications", "coding bootcamp", "language courses", "career coaching", "study planner", "job board alerts", "leadership training", "internship programs"],
    trainingQueries: ["career advice", "online education programs", "job interview tips", "professional training", "resume help", "certification courses", "how to switch careers", "internship opportunities"]
  },
  "/Law & Government": {
    tags: ["legal consultation", "small claims", "immigration law", "policy updates", "public records", "civic engagement", "tax law", "compliance audits", "government grants", "election coverage"],
    trainingQueries: ["find legal services", "public policy news", "government forms", "tax law questions", "business compliance help", "immigration attorney", "legal aid resources", "election information"]
  },
  "/Pets & Animals": {
    tags: ["pet insurance", "dog training", "cat nutrition", "aquarium supplies", "adoption centers", "grooming service", "pet boarding", "veterinary clinic", "bird care", "animal rescue"],
    trainingQueries: ["pet supplies near me", "best dog food", "cat grooming tips", "animal adoption", "veterinary services", "pet insurance plans", "dog training classes", "aquarium setup guide"]
  },
  "/Real Estate": {
    tags: ["rental listings", "mortgage preapproval", "home appraisal", "commercial office space", "property management", "open house", "first-time buyer", "real estate agent", "condo market", "investment properties"],
    trainingQueries: ["homes for sale", "rental apartments", "commercial real estate", "property investment tips", "mortgage preapproval", "find real estate agent", "housing market trends", "open house schedule"]
  },
  "/Shopping": {
    tags: ["seasonal apparel", "flash deals", "luxury accessories", "price comparisons", "sneaker drops", "retail rewards", "electronics bundles", "coupon codes", "gift guides", "online marketplace"],
    trainingQueries: ["online shopping deals", "best apparel brands", "accessories sale", "electronics discounts", "retail store near me", "coupon finder", "gift ideas", "compare product prices"]
  },
  "/Sports": {
    tags: ["team jerseys", "league standings", "home gym setup", "marathon training", "basketball shoes", "soccer analytics", "fan tickets", "fitness tracker", "tennis racket", "sports nutrition"],
    trainingQueries: ["sports team schedule", "league news", "fitness equipment", "best running shoes", "game tickets", "training for marathon", "home gym ideas", "sports supplements"]
  },
  "/Travel": {
    tags: ["budget flights", "boutique hotels", "weekend getaways", "travel insurance", "visa checklists", "guided tours", "beach resorts", "city passes", "business travel", "adventure trips"],
    trainingQueries: ["cheap flights", "hotel booking", "tourist destinations", "travel package deals", "best places to visit", "travel insurance plans", "weekend trip ideas", "international visa requirements"]
  }
};

const backendUrl = "http://127.0.0.1:8001";
const categories = Object.keys(taxonomy);
const categoryScores = Object.fromEntries(categories.map((category) => [category, 0.001]));
const priors = Object.fromEntries(categories.map((category) => [category, 1]));
const tokenWeights = {};
const tagWeights = {};
const tagCategory = {};
let searchCount = 0;
let backendOnline = null;
let latestPrediction = null;
let bubbleCleared = false;
let selectedCloudTagKey = null;
const conversionNotifications = [];
let searchBarDocked = false;
let dockAnimationPromise = null;
let cloudPointerLocal = null;

const cloudEl = document.getElementById("cloud");
const centerStageEl = document.querySelector(".center-stage");
const formEl = document.getElementById("searchForm");
const inputEl = document.getElementById("searchInput");
const statusLineEl = document.getElementById("statusLine");
const segmentPanelEl = document.getElementById("segmentPanel");
const segmentListEl = document.getElementById("segmentList");
const historyPanelEl = document.getElementById("historyPanel");
const historyListEl = document.getElementById("historyList");
const feedbackPanelEl = document.getElementById("feedbackPanel");
const feedbackPromptEl = document.getElementById("feedbackPrompt");
const feedbackUpEl = document.getElementById("feedbackUp");
const feedbackDownEl = document.getElementById("feedbackDown");
const feedbackCorrectionEl = document.getElementById("feedbackCorrection");
const feedbackCategoryEl = document.getElementById("feedbackCategory");
const feedbackSubmitEl = document.getElementById("feedbackSubmit");
const feedbackStatusEl = document.getElementById("feedbackStatus");
const clearBubbleBtnEl = document.getElementById("clearBubbleBtn");

function logStatus(message) {
    /**
     * Log status updates without rendering them in the center UI.
     * @param {string} message - Status message.
     */
    if (typeof message === "string" && message.trim()) {
        console.info(`[status] ${message}`);
    }
}

const measureCanvas = document.createElement("canvas");
const measureCtx = measureCanvas.getContext("2d");
const localHistory = [];

statusLineEl?.classList.add("hidden");

bootstrapTrainingData();
renderSegments();
renderCloud();
renderHistory();
initializeFeedbackControls();

window.addEventListener("resize", () => {
  if (searchCount > 0 && !bubbleCleared) {
    renderCloud();
  }
});

cloudEl?.addEventListener("mousemove", (event) => {
        const bounds = cloudEl.getBoundingClientRect();
        cloudPointerLocal = {
                x: event.clientX - bounds.left,
                y: event.clientY - bounds.top,
        };
        applyCursorPull();
});

cloudEl?.addEventListener("mouseleave", () => {
        cloudPointerLocal = null;
        applyCursorPull();
});

clearBubbleBtnEl?.addEventListener("click", async () => {
    /**
     * Handle the "Clear Bubble" button click event. 
     * This will erase the search history and reset the learning state.
     * @param {Event} event - The click event object.
     */
    const confirmed = window.confirm("Are you sure you'd like to delete all search history?");
    if (!confirmed) return;
    
    const erased = await eraseHistory();
    if (!erased) {
        logStatus("Could not erase history. Make sure backend is running.");
        return;
    }

    bubbleCleared = true;
    searchCount = 0;
    resetSearchBarDocking();
    cloudEl.style.opacity = "0";
    cloudEl.innerHTML = "";

    resetLearningState();
    localHistory.length = 0;
    latestPrediction = null;
    renderHistory();
    renderFeedbackPanel();
    renderSegments();
    logStatus("History erased and bubble cleared.");
});

formEl.addEventListener("submit", async (event) => {
    /**
     * Handle the search form submission event. This will process the user's search query,
     * update the learning model, and refresh the UI with new predictions and visualizations.
     * @param {Event} event - The form submission event object.
     */
    event.preventDefault();
    const query = inputEl.value.trim();
    if (!query) return;

    inputEl.value = "";
    searchCount += 1;
    bubbleCleared = false;

    const backendResponse = await submitToBackend(query);
    if (backendResponse) {
        applyBackendLearning(query, backendResponse);
        await fetchHistory();
    } else {
        const probabilities = learnFromQueryLocal(query);
        addLocalHistory(query, probabilities);
        setLatestPrediction(query, probabilities, false);
        updateStatus(probabilities, query, false);
        renderHistory();
    }

    await ensureSearchBarDocked();
    renderCloud();
    renderSegments();
});

function ensureSearchBarDocked() {
    /**
     * Move the search shell from center to top once and resolve
     * when the glide animation has completed.
     * @returns {Promise<void>}
     */
    if (searchBarDocked || !centerStageEl) {
        searchBarDocked = true;
        return Promise.resolve();
    }

    if (dockAnimationPromise) {
        return dockAnimationPromise;
    }

    dockAnimationPromise = new Promise((resolve) => {
        let finished = false;
        const cleanup = () => {
            if (finished) return;
            finished = true;
            centerStageEl.removeEventListener("transitionend", onTransitionEnd);
            searchBarDocked = true;
            dockAnimationPromise = null;
            resolve();
        };

        const onTransitionEnd = (event) => {
            if (event.target === centerStageEl && event.propertyName === "transform") {
                cleanup();
            }
        };

        centerStageEl.addEventListener("transitionend", onTransitionEnd);
        centerStageEl.classList.add("center-stage-docked");
        window.setTimeout(cleanup, 760);
    });

    return dockAnimationPromise;
}

function resetSearchBarDocking() {
    /**
     * Reset search shell docking state to initial centered position.
     */
    searchBarDocked = false;
    dockAnimationPromise = null;
    centerStageEl?.classList.remove("center-stage-docked");
}

function applyCursorPull() {
    /**
     * Apply a slight cursor attraction transform to each visible cloud word.
     */
    const words = cloudEl?.querySelectorAll(".word") || [];
    if (!words.length) return;

    words.forEach((word) => {
        if (!cloudPointerLocal) {
            word.style.setProperty("--pull-x", "0px");
            word.style.setProperty("--pull-y", "0px");
            return;
        }

        const wordX = Number.parseFloat(word.style.left || "0");
        const wordY = Number.parseFloat(word.style.top || "0");
        const dx = cloudPointerLocal.x - wordX;
        const dy = cloudPointerLocal.y - wordY;
        const distance = Math.hypot(dx, dy) || 1;
        const influence = clamp(1 - distance / 430, 0, 1);
        const pullStrength = 8.5 * influence;

        word.style.setProperty("--pull-x", `${((dx / distance) * pullStrength).toFixed(2)}px`);
        word.style.setProperty("--pull-y", `${((dy / distance) * pullStrength * 0.9).toFixed(2)}px`);
    });
}

async function submitToBackend(query) {
    /**
     * Submit the search query to the backend for processing and learning. 
     * This function will handle communication with the backend API,
     * including error handling and response parsing.
     * @param {string} query - The search query to be submitted.
     * @returns {Promise<Object|null>} - Returns the backend response payload or null if the backend is offline or an error occurs.
     */
    if (backendOnline === false) return null;

    try {
        const response = await fetch(`${backendUrl}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query })
        });

        if (!response.ok) {
        backendOnline = false;
        return null;
        }

        const payload = await response.json();
        backendOnline = true;
        return payload;
    } catch {
        backendOnline = false;
        return null;
    }
}

function applyBackendLearning(query, payload) {
    /**
     * Apply the learning from the backend response to the local model.
     * @param {string} query - The search query that was submitted.
     * @param {Object} payload - The backend response payload containing learning data.
     */
    const probabilities = payload?.probabilities || {};

    resetCategoryScores();
    const topSegments = payload?.top_segments || [];
    if (topSegments.length) {
        topSegments.forEach((segment) => {
        if (!segment?.category || typeof segment?.score !== "number") return;
        categoryScores[segment.category] = Math.max(0.001, segment.score);
        });
    } else {
        for (const category of categories) {
        const probability = probabilities[category] ?? 0;
        categoryScores[category] += probability * 1.6;
        }
    }

    resetCloudWeights();
    const cloudWords = payload?.cloud_words || [];
    cloudWords.forEach((entry) => {
        if (!entry?.tag || !entry?.category || typeof entry?.weight !== "number") return;
        tagWeights[entry.tag] = entry.weight;
        tagCategory[entry.tag] = entry.category;
    });

    setLatestPrediction(query, probabilities, true);
    updateStatus(probabilities, query, true);
}

function initializeFeedbackControls() {
    /**
     * Initialize the feedback controls by populating the category 
     * dropdown and setting up event listeners for user interactions.
     */
    if (!feedbackCategoryEl) return;

    feedbackCategoryEl.innerHTML = "";
    categories.forEach((category) => {
        const option = document.createElement("option");
        option.value = category;
        option.textContent = category;
        feedbackCategoryEl.appendChild(option);
    });

    feedbackUpEl?.addEventListener("click", async () => {
        if (!latestPrediction) return;
        const success = await submitFeedback(latestPrediction.predictedCategory, 1.2);
        if (success) {
        feedbackStatusEl.textContent = "Thanks — positive feedback saved.";
        feedbackCorrectionEl.classList.add("hidden");
        }
    });

    feedbackDownEl?.addEventListener("click", () => {
        if (!latestPrediction) return;
        feedbackCorrectionEl.classList.remove("hidden");
        feedbackStatusEl.textContent = "Select the correct category and submit.";
    });

    feedbackSubmitEl?.addEventListener("click", async () => {
        if (!latestPrediction) return;
        const category = feedbackCategoryEl.value;
        const success = await submitFeedback(category, 1.7);
        if (success) {
        feedbackStatusEl.textContent = `Correction saved: ${category}`;
        feedbackCorrectionEl.classList.add("hidden");
        }
    });
}

function setLatestPrediction(query, probabilities, fromBackend) {
    /**
     * Set the latest prediction data structure with the given query, 
     * probabilities, and source information.
     * @param {string} query - The search query that was processed.
     * @param {Object} probabilities - An object mapping categories to their predicted probabilities.
     * @param {boolean} fromBackend - A flag indicating whether the prediction came from the backend or local learning.
     */
    const ranked = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
    const predictedCategory = ranked[0]?.[0] || "/Unknown";

    latestPrediction = {
        query,
        probabilities,
        predictedCategory,
        fromBackend
    };

    renderFeedbackPanel();
}

function renderFeedbackPanel() {
    /**
     * Render the feedback panel based on the latest prediction. 
     * If there is no latest prediction, the panel will be hidden.
     */
    if (!feedbackPanelEl) return;

    if (!latestPrediction) {
        feedbackPanelEl.classList.add("hidden");
        return;
    }

    feedbackPanelEl.classList.remove("hidden");
    feedbackCorrectionEl.classList.add("hidden");
    feedbackPromptEl.textContent = `Is this correct for ${latestPrediction.predictedCategory}?`;
    feedbackStatusEl.textContent = "";
    feedbackCategoryEl.value = latestPrediction.predictedCategory;
}

async function submitFeedback(category, confidence) {
    /**
     * Submit user feedback to the backend.
     * @param {string} category - The category that the user has confirmed or corrected to.
     * @param {number} confidence - A value indicating the strength of the feedback (e.g., 1.2 for positive, 1.7 for correction).
     * @returns {Promise<boolean>} - Returns true if the feedback was successfully submitted, false otherwise.
     */
    if (!latestPrediction) return false;

    if (backendOnline === false) {
        feedbackStatusEl.textContent = "Backend is offline. Feedback can be saved when backend is running.";
        return false;
    }

    try {
        const response = await fetch(`${backendUrl}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            query: latestPrediction.query,
            category,
            confidence
            })
        });
        if (!response.ok) {
            feedbackStatusEl.textContent = "Could not save feedback.";
            return false;
        }
        applyFeedbackToLocalHistory(latestPrediction.query, category, confidence);
        await fetchHistory();
        backendOnline = true;
        return true;
    } catch {
        backendOnline = false;
        feedbackStatusEl.textContent = "Could not reach backend for feedback.";
        return false;
    }
}

function applyFeedbackToLocalHistory(query, category, confidence) {
    /**
     * Apply feedback to the local history entry that matches the given query. 
     * This allows the feedback to be reflected in the UI immediately, 
     * even if the backend is currently offline. 
     * @param {string} query - The search query that the feedback is associated with.
     * @param {string} category - The category that the user has confirmed or corrected to.
     * @param {number} confidence - A value indicating the strength of the feedback (e.g., 1.2 for positive, 1.7 for correction).
     */
    const item = localHistory.find((entry) => entry.query === query);
    if (!item) return;

    item.feedback = {
        category,
        confidence,
        created_at: new Date().toISOString(),
    };
}

function addLocalHistory(query, probabilities) {
    /**
     * Add a new entry to the local history with the given query and predicted probabilities.
     * @param {string} query - The search query that was processed.
     * @param {Object} probabilities - An object mapping categories to their predicted probabilities.
     */
        const ranked = Object.entries(probabilities)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 3)
            .map(([category, probability]) => ({ category, probability }));

        localHistory.unshift({
            query,
            predicted_category: ranked[0]?.category || "/Unknown",
            top_categories: ranked,
            created_at: new Date().toISOString(),
        });

        if (localHistory.length > 30) {
            localHistory.pop();
        }
}

async function fetchHistory() {
    /**
     * Fetch the search history from the backend and update the local history state.
     */
    try {
        const response = await fetch(`${backendUrl}/history?limit=18`);
        if (!response.ok) return;
        const payload = await response.json();
        localHistory.length = 0;
        for (const item of payload.items || []) {
        localHistory.push(item);
        }
        renderHistory();
    } catch {
        renderHistory();
    }
}

async function eraseHistory() {
    /**
     * Send a request to the backend to erase the search history. 
     * This will clear all stored queries and feedback on the backend,
     * and reset the local history as well.
     * @return {Promise<boolean>} - Returns true if the history was successfully erased, false otherwise.
     */
    try {
        const response = await fetch(`${backendUrl}/history/clear`, {
        method: "POST",
        });

        if (!response.ok) {
        backendOnline = false;
        return false;
        }

        backendOnline = true;
        return true;
    } catch {
        backendOnline = false;
        return false;
    }
}

async function trackConversionClick(tag, category) {
    /**
     * Track a simulated conversion click event for a tag.
     * @param {string} tag - The clicked tag text.
     * @param {string} category - The category associated with the tag.
     * @returns {Promise<boolean>} - True when backend accepts the click event.
     */
    if (backendOnline === false) return false;

    try {
        const response = await fetch(`${backendUrl}/conversion/click`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ tag, category, intensity: 1.0 })
        });

        if (!response.ok) {
            backendOnline = false;
            return false;
        }

        backendOnline = true;
        return true;
    } catch {
        backendOnline = false;
        return false;
    }
}

function applyCloudSelection() {
    /**
     * Apply selected/faded visual states to cloud tags.
     */
    const words = cloudEl.querySelectorAll(".word");
    words.forEach((word) => {
        const wordKey = `${word.dataset.tag}::${word.dataset.category}`;
        const isSelected = selectedCloudTagKey && wordKey === selectedCloudTagKey;
        word.classList.toggle("word-selected", Boolean(isSelected));
        word.classList.toggle("word-faded", Boolean(selectedCloudTagKey && !isSelected));
    });
}

function addConversionNotification(message) {
    /**
     * Add a conversion-click notification to the side history panel.
     * @param {string} message - Notification message text.
     */
    conversionNotifications.unshift({
        message,
        created_at: new Date().toISOString(),
    });

    if (conversionNotifications.length > 12) {
        conversionNotifications.pop();
    }

    historyPanelEl?.classList.remove("hidden");
    logStatus(message);

    renderHistory();
}

function resetCategoryScores() {
    /**
     * Reset the category scores to their initial default values.
     */
    for (const category of categories) {
        categoryScores[category] = 0.001;
    }
}

function resetCloudWeights() {
    /**
     * Reset the cloud weights and tag-category mappings. 
     */
    for (const key of Object.keys(tagWeights)) {
        delete tagWeights[key];
    }

    for (const key of Object.keys(tagCategory)) {
        delete tagCategory[key];
    }
}

function resetLearningState() {
    /**
     * Reset the entire learning state, including category scores and cloud weights.
     */
    resetCategoryScores();
    resetCloudWeights();
}

function renderHistory() {
    /**
     * Render the search history panel with the entries from localHistory. 
     * Each entry will display the query, predicted category, top categories, 
     * and any human feedback. If there are no entries, the history panel will be hidden.
     */
    if (!historyPanelEl || !historyListEl) return;

    const toTimestamp = (value) => {
        const parsed = Date.parse(value || "");
        return Number.isFinite(parsed) ? parsed : 0;
    };

    const timeline = [
        ...conversionNotifications.map((item) => ({
            type: "conversion",
            created_at: item.created_at,
            item,
        })),
        ...localHistory.map((item) => ({
            type: "query",
            created_at: item.created_at,
            item,
        })),
    ]
        .sort((a, b) => toTimestamp(b.created_at) - toTimestamp(a.created_at))
        .slice(0, 17);

    if (!timeline.length) {
        historyPanelEl.classList.add("hidden");
        historyListEl.innerHTML = "";
        return;
    }

    historyPanelEl.classList.remove("hidden");
    historyListEl.innerHTML = "";

    timeline.forEach((entry) => {
        if (entry.type === "conversion") {
            const wrapper = document.createElement("div");
            wrapper.className = "history-item history-notification";

            const title = document.createElement("div");
            title.className = "history-prediction";
            title.textContent = "Conversion signal";

            const body = document.createElement("div");
            body.className = "history-query";
            body.textContent = entry.item.message;

            wrapper.append(title, body);
            historyListEl.appendChild(wrapper);
            return;
        }

        const item = entry.item;
        const wrapper = document.createElement("div");
        wrapper.className = "history-item";

        const query = document.createElement("div");
        query.className = "history-query";
        query.textContent = `Query: ${item.query}`;

        const prediction = document.createElement("div");
        prediction.className = "history-prediction";
        prediction.textContent = `Predicted: ${item.predicted_category}`;

        const top = document.createElement("div");
        top.className = "history-topline";
        top.textContent = (item.top_categories || [])
            .map((topEntry) => `${topEntry.category} ${Math.round((topEntry.probability || 0) * 100)}%`)
            .join(" • ");

        const feedback = document.createElement("div");
        feedback.className = "history-feedback";
        if (item.feedback?.category) {
            const isCorrection = item.feedback.category !== item.predicted_category;
            feedback.textContent = isCorrection
                ? `Human feedback: corrected to ${item.feedback.category}`
                : `Human feedback: confirmed ${item.feedback.category}`;
        } else {
            feedback.textContent = "Human feedback: none";
        }

        wrapper.append(query, prediction, top, feedback);
        historyListEl.appendChild(wrapper);
    });
}

function bootstrapTrainingData() {
    /**
     * Bootstrap the learning model with the initial training data from the taxonomy. 
     * This will populate the token weights, tag weights, and priors based on the 
     * predefined categories and tags.
     */
    for (const category of categories) {
        const { trainingQueries, tags } = taxonomy[category];

        for (const tag of tags) {
        tagWeights[tag] = (tagWeights[tag] || 0) + 0.3;
        tagCategory[tag] = category;

        for (const token of tokenize(tag)) {
            addTokenWeight(token, category, 0.6);
        }
        }

        for (const sample of trainingQueries) {
        const tokens = tokenize(sample);
        for (const token of tokens) {
            addTokenWeight(token, category, 1.2);
        }
        priors[category] += 0.05;
        }
    }
}

function addTokenWeight(token, category, amount) {
    /** 
     * Add weight to a token for a specific category. 
     * @param {string} token - The token to update.
     * @param {string} category - The category to associate with the token.
     * @param {number} amount - The amount of weight to add.
     */
    if (!tokenWeights[token]) {
        tokenWeights[token] = Object.fromEntries(categories.map((entry) => [entry, 0.06]));
    }
    tokenWeights[token][category] += amount;
}

function learnFromQueryLocal(query) {
    /**
     * Learn from a local query by updating token weights, category scores, and priors.
     * @param {string} query - The query to learn from.
     * @returns {Object} - The predicted probabilities for each category.
     */
    const tokens = tokenize(query);
    const probabilities = predictCategoryProbabilities(tokens);
    const boost = 1 + Math.log1p(tokens.length);

    for (const category of categories) {
        const probability = probabilities[category];
        categoryScores[category] += probability * boost;
        priors[category] += probability * 0.03;
    }

    for (const token of tokens) {
        for (const category of categories) {
        addTokenWeight(token, category, probabilities[category] * 0.28);
        }
    }

    for (const category of categories) {
        const probability = probabilities[category];
        if (probability < 0.02) continue;

        for (const tag of taxonomy[category].tags) {
        const overlap = lexicalOverlap(query, tag);
        const tieBreaker = hashToUnit(`${query}:${tag}`);
        const relevance = probability * (0.2 + overlap * 0.7 + tieBreaker * 0.25);
        tagWeights[tag] = (tagWeights[tag] || 0) + relevance;
        tagCategory[tag] = category;
        }
    }

    for (const token of tokens) {
        if (token.length < 4) continue;
        const topCategory = Object.entries(probabilities).sort((a, b) => b[1] - a[1])[0][0];
        tagWeights[token] = (tagWeights[token] || 0) + 0.38;
        tagCategory[token] = topCategory;
    }

    return probabilities;
}

function predictCategoryProbabilities(tokens) {
    /**
     * Predict the probabilities for each category based on the given tokens, 
     * current token weights, and priors.
     * @param {string[]} tokens - The tokens extracted from the query.
     * @returns {Object} - An object mapping categories to their predicted probabilities.
     */
    const rawScores = {};
    for (const category of categories) {
        rawScores[category] = Math.log(priors[category]);
    }

    for (const token of tokens) {
        const variants = [token, token.endsWith("s") ? token.slice(0, -1) : token];
        for (const variant of variants) {
            const distribution = tokenWeights[variant];
            if (!distribution) continue;
            for (const category of categories) {
                rawScores[category] += Math.log(distribution[category]);
            }
        }
    }

    const maxScore = Math.max(...Object.values(rawScores));
    const stabilized = {};
    let sum = 0;

    for (const category of categories) {
        const value = Math.exp(rawScores[category] - maxScore);
        stabilized[category] = value;
        sum += value;
    }

    const probabilities = {};
    for (const category of categories) {
        probabilities[category] = stabilized[category] / (sum || 1);
    }

    return probabilities;
}

function renderCloud() {
     /**
     * Render the tag cloud visualization based on the current tag weights and categories.
     */
    if (searchCount === 0 || bubbleCleared || !searchBarDocked) {
        cloudEl.style.opacity = "0";
        cloudEl.innerHTML = "";
        selectedCloudTagKey = null;
        return;
    }

    cloudEl.style.opacity = "1";
    const maxWords = Math.min(36 + searchCount * 8, 170);
    const entries = Object.entries(tagWeights)
        .filter(([tag]) => tag.length > 2)
        .sort((a, b) => b[1] - a[1])
        .slice(0, maxWords);

    const bounds = cloudEl.getBoundingClientRect();
    const searchBounds = formEl?.getBoundingClientRect();
    const segmentBounds = segmentPanelEl?.classList.contains("hidden")
        ? null
        : segmentPanelEl.getBoundingClientRect();

    const safeTop = searchBounds
        ? Math.min(bounds.height * 0.44, Math.max(106, searchBounds.bottom + 24))
        : 112;
    const safeBottom = segmentBounds
        ? Math.max(safeTop + 140, Math.min(bounds.height - 24, segmentBounds.top - 18))
        : bounds.height - 40;

    const layout = buildWordLayout(entries, bounds.width, bounds.height, safeTop, safeBottom);

    cloudEl.innerHTML = "";
    layout.forEach((item) => {
        const wordEl = document.createElement("span");
        wordEl.className = "word";
        wordEl.dataset.tag = item.tag;
        wordEl.dataset.category = item.category;
        wordEl.textContent = item.tag;
        wordEl.style.left = `${item.centerX}px`;
        wordEl.style.top = `${item.centerY}px`;
        wordEl.style.fontSize = `${item.fontSize}px`;
        wordEl.style.fontWeight = `${item.fontWeight}`;
        wordEl.style.opacity = `${item.opacity.toFixed(2)}`;
        wordEl.style.color = colorForCategory(item.category, item.tintAlpha);
        wordEl.style.setProperty("--pull-x", "0px");
        wordEl.style.setProperty("--pull-y", "0px");

        const actionsEl = document.createElement("span");
        actionsEl.className = "word-actions";
        actionsEl.addEventListener("click", (event) => {
            event.stopPropagation();
        });

        const confirmEl = document.createElement("button");
        confirmEl.type = "button";
        confirmEl.className = "word-action word-action-confirm";
        confirmEl.setAttribute("aria-label", `Confirm conversion for ${item.tag}`);
        confirmEl.textContent = "✓";
        confirmEl.addEventListener("click", async (event) => {
            event.stopPropagation();
            const tracked = await trackConversionClick(item.tag, item.category);
            const scope = tracked ? "" : " (local only)";
            addConversionNotification(`Confirmed conversion on ${item.tag} (${item.category})${scope}.`);
            selectedCloudTagKey = null;
            applyCloudSelection();
        });

        const dismissEl = document.createElement("button");
        dismissEl.type = "button";
        dismissEl.className = "word-action word-action-dismiss";
        dismissEl.setAttribute("aria-label", `Dismiss conversion for ${item.tag}`);
        dismissEl.textContent = "✕";
        dismissEl.addEventListener("click", (event) => {
            event.stopPropagation();
            selectedCloudTagKey = null;
            applyCloudSelection();
        });

        actionsEl.append(confirmEl, dismissEl);
        wordEl.appendChild(actionsEl);

        wordEl.addEventListener("click", async () => {
            const clickedKey = `${item.tag}::${item.category}`;
            selectedCloudTagKey = selectedCloudTagKey === clickedKey ? null : clickedKey;
            applyCloudSelection();
        });
        cloudEl.appendChild(wordEl);
    });

    applyCloudSelection();
    applyCursorPull();
}

function buildWordLayout(entries, width, height, safeTop, safeBottom) {
    /**
     * Build the layout for the word cloud by calculating positions, 
     * font sizes, and opacities for each tag based on their weights.
     * @param {Array} entries - An array of [tag, weight] pairs sorted by weight.
     * @param {number} width - The width of the cloud area.
     * @param {number} height - The height of the cloud area.
     * @returns {Array} - An array of layout items with tag, category, position, font size, and opacity.
     */
    if (!entries.length || !measureCtx) return [];

    const topValue = entries[0][1];
    const lowValue = entries[entries.length - 1][1];
    const spread = Math.max(0.0001, topValue - lowValue);

    const occupied = [];
    const placed = [];
    const centerX = width / 2;
    const centerY = height / 2;
    const maxOuterRadius = Math.min(width * 0.47, (safeBottom - safeTop) * 0.48);

    entries.forEach(([tag, value], index) => {
        /**
         * Calculate the layout for a single tag in the word cloud. 
         * This includes determining the font size and weight based on the tag's weight,
         * as well as positioning the tag within the cloud while avoiding overlaps.
         * @param {string} tag - The tag to be placed in the cloud.
         * @param {number} value - The weight of the tag used for sizing and opacity.
         * @param {number} index - The index of the tag in the sorted entries array.
         */
        const normalized = (value - lowValue) / spread;
        const fontSize = 13 + normalized * 37;
        const fontWeight = Math.round(380 + normalized * 450);

        measureCtx.font = `${fontWeight} ${fontSize}px Inter, system-ui, sans-serif`;
        const textWidth = measureCtx.measureText(tag).width;
        const boxW = textWidth + 8;
        const boxH = fontSize + 6;

        let placedBox = null;
        let centerPosition = null;
        const angleSeed = hashToUnit(`${tag}:angle`) * Math.PI * 2;
        const targetRadius = (1 - normalized) * maxOuterRadius;

        for (let step = 0; step < 1800; step += 1) {
        const theta = angleSeed + step * 0.31;
        const radial = Math.min(maxOuterRadius, targetRadius + step * 1.35);
        const x = centerX + Math.cos(theta) * radial;
        const y = centerY + Math.sin(theta) * radial * 0.72;

        const box = {
            x: x - boxW / 2,
            y: y - boxH / 2,
            w: boxW,
            h: boxH
        };

        if (box.x < 8 || box.y < safeTop || box.x + box.w > width - 8 || box.y + box.h > safeBottom) {
            continue;
        }

        if (!intersectsAny(box, occupied)) {
            placedBox = box;
            centerPosition = { x, y };
            break;
        }
        }

        if (!placedBox) {
            return;
        }

        occupied.push(placedBox);
        placed.push({
        tag,
        category: tagCategory[tag] || categories[0],
        centerX: centerPosition.x,
        centerY: centerPosition.y,
        fontSize,
        fontWeight,
        opacity: 0.46 + normalized * 0.54,
        tintAlpha: 0.34 + normalized * 0.62
        });
    });

    return placed;
    }

    function intersectsAny(box, boxes) {
        /**
         * Check if the given box intersects with any of the boxes in the provided array.
         * @param {Object} box - The box to check for intersections, with properties x, y, w, h.
         * @param {Array} boxes - An array of boxes to check against, each with properties x, y, w, h.
         * @returns {boolean} - Returns true if the box intersects with any of the boxes in the array, false otherwise.
         */
        for (const other of boxes) {
            if (
            box.x < other.x + other.w &&
            box.x + box.w > other.x &&
            box.y < other.y + other.h &&
            box.y + box.h > other.y
            ) {
            return true;
            }
        }

    return false;
}

function renderSegments() {
    /**
     * Render the segment panel with the top categories and their corresponding scores.
     */
    if (searchCount === 0) {
        segmentPanelEl.classList.add("hidden");
        return;
    }

    segmentPanelEl.classList.remove("hidden");
    const sorted = sortedCategories();
    const topCategories = sorted.slice(0, 6);
    const total = sorted.reduce((sum, [, value]) => sum + value, 0) || 1;

    segmentListEl.innerHTML = "";
    topCategories.forEach(([category, score]) => {
        const probability = score / total;

        const row = document.createElement("div");
        row.className = "segment-row";

        const name = document.createElement("div");
        name.className = "segment-name";
        name.textContent = category;

        const track = document.createElement("div");
        track.className = "segment-track";

        const fill = document.createElement("div");
        fill.className = "segment-fill";
        fill.style.width = `${Math.max(5, probability * 100)}%`;
        fill.style.background = colorForCategory(category, 0.72);
        track.appendChild(fill);

        const value = document.createElement("div");
        value.className = "segment-score";
        value.textContent = `${Math.round(probability * 100)}%`;

        row.append(name, track, value);
        segmentListEl.appendChild(row);
    });
}

function sortedCategories() {
    /**
     * Get the categories sorted by their current scores in descending order.
     * @returns {Array} - An array of [category, score] pairs sorted by score.
     */
    return Object.entries(categoryScores).sort((a, b) => b[1] - a[1]);
}

function updateStatus(probabilities, query, fromBackend) {
    /**
     * Update the status line with the top predicted category and its probability,
     * along with an indication of whether the prediction came from the backend or local learning.
     * @param {Object} probabilities - An object mapping categories to their predicted probabilities.
     * @param {string} query - The search query that was processed.
     * @param {boolean} fromBackend - A flag indicating whether the prediction came from the backend or local learning.
     */
    const sorted = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
    const [topCategory, topProbability] = sorted[0] || ["/Unknown", 0];

    if (fromBackend) {
        logStatus(`Backend model learned from "${query}" → strongest segment: ${topCategory} (${Math.round(topProbability * 100)}%)`);
    } else {
        logStatus(`Local model learned from "${query}" → strongest segment: ${topCategory} (${Math.round(topProbability * 100)}%)`);
    }
    }

    function lexicalOverlap(query, tag) {
        /**
         * Calculate the lexical overlap between the search query and a tag.
         * @param {string} query - The search query to compare.
         * @param {string} tag - The tag to compare against the query.
         * @returns {number} - A value between 0 and 1 representing the lexical overlap.
         */
    const queryTokens = new Set(tokenize(query));
    const tagTokens = tokenize(tag);
    if (!tagTokens.length) return 0;

    let overlap = 0;
    for (const token of tagTokens) {
        if (queryTokens.has(token)) overlap += 1;
    }

    return overlap / tagTokens.length;
    }

    function tokenize(text) {
        /**
         * Tokenize the input text by converting it to lowercase, removing non-alphanumeric characters,
         * splitting on whitespace, and filtering out short tokens.
         * @param {string} text - The text to tokenize.
         * @returns {Array} - An array of tokens extracted from the input text.
         */
    return text
        .toLowerCase()
        .replace(/[^a-z0-9\s]/g, " ")
        .split(/\s+/)
        .filter((token) => token.length > 1);
    }

    function hashToUnit(value) {
        /**
         * Generate a deterministic hash value between 0 and 1 for the given input string using the FNV-1a hashing algorithm.
         * @param {string} value - The input string to hash.
         * @returns {number} - A deterministic value between 0 and 1 derived from the input string.
         */
    let hash = 2166136261;
    for (let index = 0; index < value.length; index += 1) {
        hash ^= value.charCodeAt(index);
        hash = Math.imul(hash, 16777619);
    }

    return ((hash >>> 0) % 1000) / 1000;
    }

    function clamp(value, min, max) {
        /**
         * Clamp a value between a minimum and maximum range.
         * @param {number} value - The value to clamp.
         * @param {number} min - The minimum allowable value.
         * @param {number} max - The maximum allowable value.
         * @returns {number} - The clamped value.
         */
        return Math.max(min, Math.min(max, value));
    }

    function colorForCategory(category, alpha) {
        /**
         * Get the color for a given category with the specified alpha transparency.
         * The colors are defined in the palette object, which maps categories to HSL color values.
         * If the category is not found in the palette, a default gray color is used.
         * @param {string} category - The category for which to get the color.
         * @param {number} alpha - The alpha transparency value (between 0 and 1) for the color.
         * @returns {string} - An HSLA color string representing the color for the category with the specified alpha.
         */
    const palette = {
        "/Arts & Entertainment": [210, 73, 53],
        "/Autos & Vehicles": [14, 68, 51],
        "/Beauty & Fitness": [332, 68, 55],
        "/Books & Literature": [266, 48, 46],
        "/Business & Industrial": [20, 57, 44],
        "/Computers & Electronics": [226, 60, 52],
        "/Finance": [154, 49, 42],
        "/Food & Drink": [32, 72, 52],
        "/Games": [274, 70, 56],
        "/Health": [186, 50, 44],
        "/Hobbies & Leisure": [103, 42, 40],
        "/Home & Garden": [84, 38, 40],
        "/Internet & Telecom": [196, 60, 48],
        "/Jobs & Education": [46, 60, 44],
        "/Law & Government": [8, 32, 37],
        "/Pets & Animals": [128, 40, 43],
        "/Real Estate": [48, 44, 44],
        "/Shopping": [316, 62, 52],
        "/Sports": [358, 58, 45],
        "/Travel": [191, 58, 47]
    };

    const [h, s, l] = palette[category] || [0, 0, 20];
    return `hsla(${h}, ${s}%, ${l}%, ${alpha})`;
}
