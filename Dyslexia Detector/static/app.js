/* ==========================================================================
   app.js
   JavaScript utilities for Frontend Projects
   (E-Commerce, Food Delivery, Grocery, Museum & Zoo Ticketing sites)

   JSDoc type annotations are used throughout so editors (VS Code, WebStorm)
   still provide type-checking and autocomplete without a build step.

   Modules:
   1.  Cart
   2.  Product filter, sort & grid rendering
   3.  Form validation (booking / checkout)
   4.  Ticket / visitor counter widget
   5.  Debounce, throttle & event delegation utilities
   6.  Wishlist
   7.  Pagination
   8.  Star rating / review system
   9.  Coupon / discount engine
   10. Toast notification system
   11. Modal manager
   12. Lazy-loading images (IntersectionObserver)
   13. Mock authentication (localStorage based)
   14. Order history
   15. Page wiring on DOMContentLoaded
   ========================================================================== */

"use strict";

/**
 * @typedef {Object} Product
 * @property {string} id
 * @property {string} name
 * @property {number} price
 * @property {string} category
 * @property {string} image
 */

/**
 * @typedef {Product & { quantity: number }} CartItem
 */

/**
 * @typedef {Object} BookingFormData
 * @property {string} name
 * @property {string} email
 * @property {string} phone
 * @property {string} date
 * @property {string} visitors
 */

/**
 * @typedef {Object} ValidationResult
 * @property {boolean} isValid
 * @property {Object.<string, string>} errors
 */

/**
 * @typedef {Object} FilterOptions
 * @property {string} [category]
 * @property {string} [keyword]
 */

/**
 * @typedef {Object} Review
 * @property {string} id
 * @property {string} productId
 * @property {string} author
 * @property {number} rating
 * @property {string} comment
 * @property {string} date
 */

/**
 * @typedef {Object} Coupon
 * @property {string} code
 * @property {"percent"|"flat"} type
 * @property {number} value
 * @property {number} [minOrderValue]
 * @property {string} [expiry]
 */

/**
 * @typedef {Object} CouponResult
 * @property {boolean} valid
 * @property {string} message
 * @property {number} discountAmount
 */

/**
 * @typedef {"success"|"error"|"info"|"warning"} ToastType
 */

/**
 * @typedef {Object} UserAccount
 * @property {string} email
 * @property {string} passwordHash
 * @property {string} name
 */

/**
 * @typedef {Object} Order
 * @property {string} id
 * @property {CartItem[]} items
 * @property {number} total
 * @property {string} placedAt
 * @property {"pending"|"confirmed"|"delivered"|"cancelled"} status
 */

/* --------------------------------------------------------------------------
   1. CART MODULE
   -------------------------------------------------------------------------- */
const Cart = (() => {
  const STORAGE_KEY = "site_cart_items";

  /** @returns {CartItem[]} */
  function getItems() {
    const raw = localStorage.getItem(STORAGE_KEY);
    try {
      return raw ? JSON.parse(raw) : [];
    } catch (err) {
      console.error("Cart parse error:", err);
      return [];
    }
  }

  /** @param {CartItem[]} items */
  function saveItems(items) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(items));
    updateCartBadge();
  }

  /**
   * @param {Product} product
   * @param {number} [quantity]
   * @returns {CartItem[]}
   */
  function addItem(product, quantity = 1) {
    const items = getItems();
    const existing = items.find((item) => item.id === product.id);

    if (existing) {
      existing.quantity += quantity;
    } else {
      items.push({ ...product, quantity });
    }

    saveItems(items);
    return items;
  }

  /**
   * @param {string} productId
   * @returns {CartItem[]}
   */
  function removeItem(productId) {
    const items = getItems().filter((item) => item.id !== productId);
    saveItems(items);
    return items;
  }

  /**
   * @param {string} productId
   * @param {number} quantity
   * @returns {CartItem[]}
   */
  function updateQuantity(productId, quantity) {
    const items = getItems();
    const item = items.find((i) => i.id === productId);
    if (item) {
      item.quantity = Math.max(1, Number(quantity));
    }
    saveItems(items);
    return items;
  }

  function clearCart() {
    saveItems([]);
  }

  /** @returns {number} */
  function getTotal() {
    return getItems().reduce((sum, item) => sum + item.price * item.quantity, 0);
  }

  /** @returns {number} */
  function getItemCount() {
    return getItems().reduce((sum, item) => sum + item.quantity, 0);
  }

  function updateCartBadge() {
    const badge = document.querySelector("[data-cart-count]");
    if (badge) {
      badge.textContent = String(getItemCount());
    }
  }

  return {
    getItems,
    addItem,
    removeItem,
    updateQuantity,
    clearCart,
    getTotal,
    getItemCount,
    updateCartBadge,
  };
})();

/* --------------------------------------------------------------------------
   2. PRODUCT / MENU FILTER MODULE
   -------------------------------------------------------------------------- */

/**
 * @param {Product[]} products
 * @param {FilterOptions} [options]
 * @returns {Product[]}
 */
function filterProducts(products, { category = "all", keyword = "" } = {}) {
  const normalizedKeyword = keyword.trim().toLowerCase();

  return products.filter((product) => {
    const matchesCategory =
      category === "all" || product.category.toLowerCase() === category.toLowerCase();

    const matchesKeyword =
      normalizedKeyword === "" || product.name.toLowerCase().includes(normalizedKeyword);

    return matchesCategory && matchesKeyword;
  });
}

/**
 * @param {Product[]} products
 * @param {"price-asc"|"price-desc"|"name-asc"|"name-desc"} sortBy
 * @returns {Product[]}
 */
function sortProducts(products, sortBy) {
  const sorted = [...products];
  switch (sortBy) {
    case "price-asc":
      return sorted.sort((a, b) => a.price - b.price);
    case "price-desc":
      return sorted.sort((a, b) => b.price - a.price);
    case "name-asc":
      return sorted.sort((a, b) => a.name.localeCompare(b.name));
    case "name-desc":
      return sorted.sort((a, b) => b.name.localeCompare(a.name));
    default:
      return sorted;
  }
}

/**
 * @param {Product[]} products
 * @param {string} containerSelector
 */
function renderProductGrid(products, containerSelector) {
  const container = document.querySelector(containerSelector);
  if (!container) return;

  if (products.length === 0) {
    container.innerHTML = `<p class="text-center text-muted">No items found.</p>`;
    return;
  }

  container.innerHTML = products
    .map(
      (product) => `
      <div class="col-md-4 col-sm-6 mb-4 product-card" data-id="${product.id}">
        <div class="card h-100 shadow-sm">
          <img data-src="${product.image}" class="card-img-top lazy-img" alt="${product.name}">
          <div class="card-body">
            <h5 class="card-title">${product.name}</h5>
            <p class="card-text">₹${product.price.toFixed(2)}</p>
            <button class="btn btn-outline-danger btn-sm wishlist-btn" data-id="${product.id}">♥</button>
            <button class="btn btn-primary btn-sm add-to-cart-btn" data-id="${product.id}">
              Add to Cart
            </button>
          </div>
        </div>
      </div>`
    )
    .join("");

  initLazyLoad(".lazy-img");
}

/* --------------------------------------------------------------------------
   3. FORM VALIDATION MODULE (booking / checkout forms)
   -------------------------------------------------------------------------- */
const Validator = (() => {
  /** @param {string} value @returns {boolean} */
  function isEmpty(value) {
    return !value || value.trim().length === 0;
  }

  /** @param {string} email @returns {boolean} */
  function isValidEmail(email) {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
  }

  /** @param {string} phone @returns {boolean} */
  function isValidPhone(phone) {
    return /^[0-9]{10}$/.test(phone.trim());
  }

  /** @param {string} dateStr @returns {boolean} */
  function isFutureDate(dateStr) {
    const inputDate = new Date(dateStr);
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    return inputDate >= today;
  }

  /** @param {string} value @returns {boolean} */
  function isPositiveInteger(value) {
    const num = Number(value);
    return Number.isInteger(num) && num > 0;
  }

  /**
   * @param {BookingFormData} formData
   * @returns {ValidationResult}
   */
  function validateBookingForm(formData) {
    const errors = {};

    if (isEmpty(formData.name)) errors.name = "Name is required.";
    if (!isValidEmail(formData.email)) errors.email = "Enter a valid email address.";
    if (!isValidPhone(formData.phone)) errors.phone = "Enter a valid 10-digit phone number.";
    if (isEmpty(formData.date) || !isFutureDate(formData.date)) {
      errors.date = "Please select a valid future date.";
    }
    if (!isPositiveInteger(formData.visitors)) {
      errors.visitors = "Number of visitors must be a positive number.";
    }

    return { isValid: Object.keys(errors).length === 0, errors };
  }

  /** @param {Object.<string, string>} errors */
  function showErrors(errors) {
    document.querySelectorAll(".error-message").forEach((el) => (el.textContent = ""));
    Object.entries(errors).forEach(([field, message]) => {
      const el = document.querySelector(`[data-error-for="${field}"]`);
      if (el) el.textContent = message;
    });
  }

  return { validateBookingForm, showErrors, isValidEmail, isValidPhone, isFutureDate };
})();

/* --------------------------------------------------------------------------
   4. TICKET / VISITOR COUNTER WIDGET (Museum & Zoo ticketing)
   -------------------------------------------------------------------------- */

/**
 * @param {string} counterSelector
 * @param {number} pricePerTicket
 */
function initVisitorCounter(counterSelector, pricePerTicket) {
  const container = document.querySelector(counterSelector);
  if (!container) return;

  const decrementBtn = container.querySelector("[data-action='decrement']");
  const incrementBtn = container.querySelector("[data-action='increment']");
  const countDisplay = container.querySelector("[data-visitor-count]");
  const totalDisplay = container.querySelector("[data-total-price]");

  let count = 1;

  function render() {
    if (countDisplay) countDisplay.textContent = String(count);
    if (totalDisplay) totalDisplay.textContent = `₹${(count * pricePerTicket).toFixed(2)}`;
  }

  decrementBtn?.addEventListener("click", () => {
    count = Math.max(1, count - 1);
    render();
  });

  incrementBtn?.addEventListener("click", () => {
    count += 1;
    render();
  });

  render();
}

/* --------------------------------------------------------------------------
   5. UTILITIES: debounce, throttle, event delegation
   -------------------------------------------------------------------------- */

/**
 * @template {(...args: any[]) => void} T
 * @param {T} fn
 * @param {number} [delay]
 * @returns {(...args: Parameters<T>) => void}
 */
function debounce(fn, delay = 300) {
  let timeoutId;
  return function debounced(...args) {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => fn.apply(this, args), delay);
  };
}

/**
 * @template {(...args: any[]) => void} T
 * @param {T} fn
 * @param {number} [limit]
 * @returns {(...args: Parameters<T>) => void}
 */
function throttle(fn, limit = 300) {
  let waiting = false;
  return function throttled(...args) {
    if (!waiting) {
      fn.apply(this, args);
      waiting = true;
      setTimeout(() => (waiting = false), limit);
    }
  };
}

/**
 * @param {string} parentSelector
 * @param {string} eventType
 * @param {string} childSelector
 * @param {(event: Event, target: HTMLElement) => void} handler
 */
function delegateEvent(parentSelector, eventType, childSelector, handler) {
  const parent = document.querySelector(parentSelector);
  if (!parent) return;

  parent.addEventListener(eventType, (event) => {
    const target = event.target.closest(childSelector);
    if (target && parent.contains(target)) {
      handler(event, target);
    }
  });
}

/* --------------------------------------------------------------------------
   6. WISHLIST MODULE
   -------------------------------------------------------------------------- */
const Wishlist = (() => {
  const STORAGE_KEY = "site_wishlist_items";

  /** @returns {Product[]} */
  function getItems() {
    const raw = localStorage.getItem(STORAGE_KEY);
    try {
      return raw ? JSON.parse(raw) : [];
    } catch {
      return [];
    }
  }

  /** @param {Product[]} items */
  function save(items) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(items));
    updateBadge();
  }

  /**
   * @param {Product} product
   * @returns {boolean} true if added, false if removed
   */
  function toggle(product) {
    const items = getItems();
    const index = items.findIndex((p) => p.id === product.id);
    let added;

    if (index >= 0) {
      items.splice(index, 1);
      added = false;
    } else {
      items.push(product);
      added = true;
    }

    save(items);
    return added;
  }

  /** @param {string} productId @returns {boolean} */
  function isWishlisted(productId) {
    return getItems().some((p) => p.id === productId);
  }

  function updateBadge() {
    const badge = document.querySelector("[data-wishlist-count]");
    if (badge) badge.textContent = String(getItems().length);
  }

  return { getItems, toggle, isWishlisted, updateBadge };
})();

/* --------------------------------------------------------------------------
   7. PAGINATION MODULE
   -------------------------------------------------------------------------- */
class Paginator {
  /**
   * @param {any[]} items
   * @param {number} [pageSize]
   */
  constructor(items, pageSize = 6) {
    this.items = items;
    this.pageSize = pageSize;
    this.currentPage = 1;
  }

  /** @returns {number} */
  get totalPages() {
    return Math.max(1, Math.ceil(this.items.length / this.pageSize));
  }

  /**
   * @param {number} page
   * @returns {any[]}
   */
  getPage(page) {
    this.currentPage = Math.min(Math.max(1, page), this.totalPages);
    const start = (this.currentPage - 1) * this.pageSize;
    return this.items.slice(start, start + this.pageSize);
  }

  /** @returns {any[]} */
  next() {
    return this.getPage(this.currentPage + 1);
  }

  /** @returns {any[]} */
  prev() {
    return this.getPage(this.currentPage - 1);
  }

  /** @returns {number} */
  getCurrentPage() {
    return this.currentPage;
  }

  /** @param {any[]} items */
  updateItems(items) {
    this.items = items;
    this.currentPage = 1;
  }

  /**
   * @param {string} containerSelector
   * @param {(page: number) => void} onPageChange
   */
  renderControls(containerSelector, onPageChange) {
    const container = document.querySelector(containerSelector);
    if (!container) return;

    const buttons = [];
    for (let i = 1; i <= this.totalPages; i++) {
      buttons.push(
        `<button class="btn btn-sm ${i === this.currentPage ? "btn-primary" : "btn-outline-secondary"} page-btn" data-page="${i}">${i}</button>`
      );
    }
    container.innerHTML = buttons.join(" ");

    container.querySelectorAll(".page-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const page = Number(btn.getAttribute("data-page"));
        onPageChange(page);
        this.renderControls(containerSelector, onPageChange);
      });
    });
  }
}

/* --------------------------------------------------------------------------
   8. STAR RATING / REVIEW SYSTEM
   -------------------------------------------------------------------------- */
const Reviews = (() => {
  const STORAGE_KEY = "site_product_reviews";

  /** @returns {Review[]} */
  function getAll() {
    const raw = localStorage.getItem(STORAGE_KEY);
    try {
      return raw ? JSON.parse(raw) : [];
    } catch {
      return [];
    }
  }

  /** @param {string} productId @returns {Review[]} */
  function getForProduct(productId) {
    return getAll().filter((r) => r.productId === productId);
  }

  /**
   * @param {Omit<Review, "id"|"date">} review
   * @returns {Review}
   */
  function addReview(review) {
    const newReview = {
      ...review,
      id: `rev_${Date.now()}_${Math.floor(Math.random() * 1000)}`,
      date: new Date().toISOString().split("T")[0],
    };
    const all = getAll();
    all.push(newReview);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(all));
    return newReview;
  }

  /** @param {string} productId @returns {number} */
  function getAverageRating(productId) {
    const reviews = getForProduct(productId);
    if (reviews.length === 0) return 0;
    const total = reviews.reduce((sum, r) => sum + r.rating, 0);
    return Math.round((total / reviews.length) * 10) / 10;
  }

  /** @param {number} rating @returns {string} */
  function renderStars(rating) {
    const fullStars = Math.floor(rating);
    const hasHalf = rating - fullStars >= 0.5;
    let html = "";
    for (let i = 0; i < fullStars; i++) html += "★";
    if (hasHalf) html += "☆";
    const empty = 5 - fullStars - (hasHalf ? 1 : 0);
    for (let i = 0; i < empty; i++) html += "✩";
    return html;
  }

  return { getAll, getForProduct, addReview, getAverageRating, renderStars };
})();

/* --------------------------------------------------------------------------
   9. COUPON / DISCOUNT ENGINE
   -------------------------------------------------------------------------- */
const CouponEngine = (() => {
  /** @type {Coupon[]} */
  const AVAILABLE_COUPONS = [
    { code: "SAVE10", type: "percent", value: 10, minOrderValue: 500 },
    { code: "FLAT50", type: "flat", value: 50, minOrderValue: 300 },
    { code: "WELCOME20", type: "percent", value: 20, minOrderValue: 1000, expiry: "2026-12-31" },
  ];

  /** @param {Coupon} coupon @returns {boolean} */
  function isExpired(coupon) {
    if (!coupon.expiry) return false;
    return new Date(coupon.expiry) < new Date();
  }

  /**
   * @param {string} code
   * @param {number} orderTotal
   * @returns {CouponResult}
   */
  function applyCoupon(code, orderTotal) {
    const coupon = AVAILABLE_COUPONS.find((c) => c.code.toLowerCase() === code.trim().toLowerCase());

    if (!coupon) {
      return { valid: false, message: "Invalid coupon code.", discountAmount: 0 };
    }
    if (isExpired(coupon)) {
      return { valid: false, message: "This coupon has expired.", discountAmount: 0 };
    }
    if (coupon.minOrderValue && orderTotal < coupon.minOrderValue) {
      return {
        valid: false,
        message: `Minimum order value of ₹${coupon.minOrderValue} required.`,
        discountAmount: 0,
      };
    }

    const discountAmount =
      coupon.type === "percent" ? (orderTotal * coupon.value) / 100 : coupon.value;

    return {
      valid: true,
      message: `Coupon applied! You saved ₹${discountAmount.toFixed(2)}.`,
      discountAmount,
    };
  }

  return { applyCoupon, AVAILABLE_COUPONS };
})();

/* --------------------------------------------------------------------------
   10. TOAST NOTIFICATION SYSTEM
   -------------------------------------------------------------------------- */
const Toast = (() => {
  /** @type {HTMLElement|null} */
  let container = null;

  /** @returns {HTMLElement} */
  function ensureContainer() {
    if (container) return container;
    container = document.createElement("div");
    container.setAttribute("data-toast-container", "");
    container.style.position = "fixed";
    container.style.bottom = "20px";
    container.style.right = "20px";
    container.style.zIndex = "9999";
    document.body.appendChild(container);
    return container;
  }

  /** @param {ToastType} type @returns {string} */
  function colorFor(type) {
    switch (type) {
      case "success":
        return "#198754";
      case "error":
        return "#dc3545";
      case "warning":
        return "#ffc107";
      default:
        return "#0dcaf0";
    }
  }

  /**
   * @param {string} message
   * @param {ToastType} [type]
   * @param {number} [duration]
   */
  function show(message, type = "info", duration = 3000) {
    const root = ensureContainer();
    const toast = document.createElement("div");
    toast.textContent = message;
    toast.style.background = colorFor(type);
    toast.style.color = "#fff";
    toast.style.padding = "10px 16px";
    toast.style.marginTop = "8px";
    toast.style.borderRadius = "6px";
    toast.style.boxShadow = "0 2px 6px rgba(0,0,0,0.2)";
    toast.style.opacity = "0";
    toast.style.transition = "opacity 0.3s ease";

    root.appendChild(toast);
    requestAnimationFrame(() => (toast.style.opacity = "1"));

    setTimeout(() => {
      toast.style.opacity = "0";
      setTimeout(() => toast.remove(), 300);
    }, duration);
  }

  return { show };
})();

/* --------------------------------------------------------------------------
   11. MODAL MANAGER
   -------------------------------------------------------------------------- */
const ModalManager = (() => {
  /** @param {string} modalSelector */
  function open(modalSelector) {
    const modal = document.querySelector(modalSelector);
    if (!modal) return;
    modal.classList.add("show");
    modal.style.display = "block";
    document.body.classList.add("modal-open");
  }

  /** @param {string} modalSelector */
  function close(modalSelector) {
    const modal = document.querySelector(modalSelector);
    if (!modal) return;
    modal.classList.remove("show");
    modal.style.display = "none";
    document.body.classList.remove("modal-open");
  }

  /**
   * @param {string} modalSelector
   * @param {string} closeTriggerSelector
   */
  function bindCloseTriggers(modalSelector, closeTriggerSelector) {
    document.querySelectorAll(closeTriggerSelector).forEach((trigger) => {
      trigger.addEventListener("click", () => close(modalSelector));
    });
  }

  return { open, close, bindCloseTriggers };
})();

/* --------------------------------------------------------------------------
   12. LAZY-LOADING IMAGES (IntersectionObserver)
   -------------------------------------------------------------------------- */

/** @param {string} imageSelector */
function initLazyLoad(imageSelector) {
  const images = document.querySelectorAll(imageSelector);
  if (images.length === 0) return;

  if (!("IntersectionObserver" in window)) {
    images.forEach((img) => {
      const src = img.getAttribute("data-src");
      if (src) img.src = src;
    });
    return;
  }

  const observer = new IntersectionObserver(
    (entries, obs) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          const img = entry.target;
          const src = img.getAttribute("data-src");
          if (src) img.src = src;
          img.classList.add("loaded");
          obs.unobserve(img);
        }
      });
    },
    { rootMargin: "50px" }
  );

  images.forEach((img) => observer.observe(img));
}

/* --------------------------------------------------------------------------
   13. MOCK AUTHENTICATION (localStorage based, demo purposes only)
   -------------------------------------------------------------------------- */
const Auth = (() => {
  const USERS_KEY = "site_users";
  const SESSION_KEY = "site_session";

  /** @param {string} value @returns {string} */
  function simpleHash(value) {
    let hash = 0;
    for (let i = 0; i < value.length; i++) {
      hash = (hash << 5) - hash + value.charCodeAt(i);
      hash |= 0;
    }
    return String(hash);
  }

  /** @returns {UserAccount[]} */
  function getUsers() {
    const raw = localStorage.getItem(USERS_KEY);
    try {
      return raw ? JSON.parse(raw) : [];
    } catch {
      return [];
    }
  }

  /**
   * @param {string} name
   * @param {string} email
   * @param {string} password
   * @returns {{success: boolean, message: string}}
   */
  function register(name, email, password) {
    const users = getUsers();
    if (users.some((u) => u.email.toLowerCase() === email.toLowerCase())) {
      return { success: false, message: "An account with this email already exists." };
    }
    users.push({ name, email, passwordHash: simpleHash(password) });
    localStorage.setItem(USERS_KEY, JSON.stringify(users));
    return { success: true, message: "Registration successful. Please log in." };
  }

  /**
   * @param {string} email
   * @param {string} password
   * @returns {{success: boolean, message: string}}
   */
  function login(email, password) {
    const users = getUsers();
    const user = users.find((u) => u.email.toLowerCase() === email.toLowerCase());

    if (!user || user.passwordHash !== simpleHash(password)) {
      return { success: false, message: "Invalid email or password." };
    }

    localStorage.setItem(SESSION_KEY, JSON.stringify({ email: user.email, name: user.name }));
    return { success: true, message: `Welcome back, ${user.name}!` };
  }

  function logout() {
    localStorage.removeItem(SESSION_KEY);
  }

  /** @returns {{email: string, name: string}|null} */
  function getCurrentUser() {
    const raw = localStorage.getItem(SESSION_KEY);
    try {
      return raw ? JSON.parse(raw) : null;
    } catch {
      return null;
    }
  }

  /** @returns {boolean} */
  function isLoggedIn() {
    return getCurrentUser() !== null;
  }

  return { register, login, logout, getCurrentUser, isLoggedIn };
})();

/* --------------------------------------------------------------------------
   14. ORDER HISTORY
   -------------------------------------------------------------------------- */
const OrderHistory = (() => {
  const STORAGE_KEY = "site_order_history";

  /** @returns {Order[]} */
  function getAll() {
    const raw = localStorage.getItem(STORAGE_KEY);
    try {
      return raw ? JSON.parse(raw) : [];
    } catch {
      return [];
    }
  }

  /**
   * @param {CartItem[]} items
   * @returns {Order}
   */
  function placeOrder(items) {
    const order = {
      id: `ORD${Date.now()}`,
      items,
      total: items.reduce((sum, item) => sum + item.price * item.quantity, 0),
      placedAt: new Date().toISOString(),
      status: "pending",
    };
    const all = getAll();
    all.unshift(order);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(all));
    return order;
  }

  /**
   * @param {string} orderId
   * @param {Order["status"]} status
   */
  function updateStatus(orderId, status) {
    const all = getAll();
    const order = all.find((o) => o.id === orderId);
    if (order) order.status = status;
    localStorage.setItem(STORAGE_KEY, JSON.stringify(all));
  }

  /** @param {string} containerSelector */
  function renderHistory(containerSelector) {
    const container = document.querySelector(containerSelector);
    if (!container) return;

    const orders = getAll();
    if (orders.length === 0) {
      container.innerHTML = `<p class="text-muted">No orders placed yet.</p>`;
      return;
    }

    container.innerHTML = orders
      .map(
        (order) => `
        <div class="card mb-3 p-3">
          <div class="d-flex justify-content-between">
            <strong>${order.id}</strong>
            <span class="badge bg-secondary">${order.status}</span>
          </div>
          <small class="text-muted">${new Date(order.placedAt).toLocaleString()}</small>
          <p class="mb-0">Total: ₹${order.total.toFixed(2)} • ${order.items.length} item(s)</p>
        </div>`
      )
      .join("");
  }

  return { getAll, placeOrder, updateStatus, renderHistory };
})();

/* --------------------------------------------------------------------------
   15. WIRING IT TOGETHER ON PAGE LOAD
   -------------------------------------------------------------------------- */
document.addEventListener("DOMContentLoaded", () => {
  Cart.updateCartBadge();
  Wishlist.updateBadge();
  initLazyLoad(".lazy-img");

  const searchInput = document.querySelector("[data-search-input]");
  const categoryFilter = document.querySelector("[data-category-filter]");
  const sortSelect = document.querySelector("[data-sort-select]");

  function refreshGrid() {
    if (!window.PRODUCTS) return;
    let result = filterProducts(window.PRODUCTS, {
      keyword: searchInput?.value ?? "",
      category: categoryFilter?.value ?? "all",
    });
    if (sortSelect?.value) {
      result = sortProducts(result, sortSelect.value);
    }
    renderProductGrid(result, "[data-product-grid]");
  }

  searchInput?.addEventListener("input", debounce(refreshGrid, 250));
  categoryFilter?.addEventListener("change", refreshGrid);
  sortSelect?.addEventListener("change", refreshGrid);

  // Delegated "Add to Cart" clicks
  delegateEvent("[data-product-grid]", "click", ".add-to-cart-btn", (_event, target) => {
    const id = target.getAttribute("data-id");
    const product = window.PRODUCTS?.find((p) => p.id === id);
    if (product) {
      Cart.addItem(product);
      Toast.show(`${product.name} added to cart`, "success");
      target.textContent = "Added ✓";
      setTimeout(() => (target.textContent = "Add to Cart"), 1000);
    }
  });

  // Delegated wishlist toggle
  delegateEvent("[data-product-grid]", "click", ".wishlist-btn", (_event, target) => {
    const id = target.getAttribute("data-id");
    const product = window.PRODUCTS?.find((p) => p.id === id);
    if (product) {
      const added = Wishlist.toggle(product);
      Toast.show(added ? "Added to wishlist" : "Removed from wishlist", "info");
    }
  });

  // Coupon form
  const couponForm = document.querySelector("#couponForm");
  if (couponForm) {
    couponForm.addEventListener("submit", (e) => {
      e.preventDefault();
      const input = couponForm.querySelector("[name='coupon']");
      if (!input) return;
      const result = CouponEngine.applyCoupon(input.value, Cart.getTotal());
      Toast.show(result.message, result.valid ? "success" : "error");
    });
  }

  // Booking form submission
  const bookingForm = document.querySelector("#bookingForm");
  if (bookingForm) {
    bookingForm.addEventListener("submit", (e) => {
      e.preventDefault();
      const formData = Object.fromEntries(new FormData(bookingForm).entries());
      const { isValid, errors } = Validator.validateBookingForm(formData);

      Validator.showErrors(errors);

      if (isValid) {
        Toast.show("Booking confirmed! Thank you.", "success");
        bookingForm.reset();
      }
    });
  }

  // Review form
  const reviewForm = document.querySelector("#reviewForm");
  if (reviewForm) {
    reviewForm.addEventListener("submit", (e) => {
      e.preventDefault();
      const data = Object.fromEntries(new FormData(reviewForm).entries());
      Reviews.addReview({
        productId: data.productId,
        author: data.author,
        rating: Number(data.rating),
        comment: data.comment,
      });
      Toast.show("Review submitted!", "success");
      reviewForm.reset();
    });
  }

  // Login / register forms
  const loginForm = document.querySelector("#loginForm");
  loginForm?.addEventListener("submit", (e) => {
    e.preventDefault();
    const data = Object.fromEntries(new FormData(loginForm).entries());
    const result = Auth.login(data.email, data.password);
    Toast.show(result.message, result.success ? "success" : "error");
  });

  const registerForm = document.querySelector("#registerForm");
  registerForm?.addEventListener("submit", (e) => {
    e.preventDefault();
    const data = Object.fromEntries(new FormData(registerForm).entries());
    const result = Auth.register(data.name, data.email, data.password);
    Toast.show(result.message, result.success ? "success" : "error");
  });

  // Checkout button -> place order
  const checkoutBtn = document.querySelector("[data-checkout-btn]");
  checkoutBtn?.addEventListener("click", () => {
    const items = Cart.getItems();
    if (items.length === 0) {
      Toast.show("Your cart is empty.", "warning");
      return;
    }
    const order = OrderHistory.placeOrder(items);
    Cart.clearCart();
    Toast.show(`Order ${order.id} placed successfully!`, "success");
    OrderHistory.renderHistory("[data-order-history]");
  });

  // Modal bindings
  ModalManager.bindCloseTriggers("[data-modal]", "[data-modal-close]");
  document.querySelectorAll("[data-modal-open]").forEach((trigger) => {
    trigger.addEventListener("click", () => {
      const target = trigger.getAttribute("data-modal-open");
      if (target) ModalManager.open(target);
    });
  });

  // Visitor counter widget (Museum/Zoo)
  initVisitorCounter("[data-visitor-counter]", 250);

  // Order history render on load (if present)
  OrderHistory.renderHistory("[data-order-history]");
});
