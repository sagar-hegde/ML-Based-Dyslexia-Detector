/* ==========================================================================
   app.ts
   Typed TypeScript utilities for Frontend Projects
   (E-Commerce, Food Delivery, Grocery, Museum & Zoo Ticketing sites)

   Modules:
   1.  Cart
   2.  Product filter & grid rendering
   3.  Form validation (booking / checkout)
   4.  Ticket / visitor counter widget
   5.  Debounce & event delegation utilities
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

/* --------------------------------------------------------------------------
   TYPE DEFINITIONS
   -------------------------------------------------------------------------- */
interface Product {
  id: string;
  name: string;
  price: number;
  category: string;
  image: string;
}

interface CartItem extends Product {
  quantity: number;
}

interface BookingFormData {
  name: string;
  email: string;
  phone: string;
  date: string;
  visitors: string;
}

interface ValidationResult {
  isValid: boolean;
  errors: Partial<Record<keyof BookingFormData, string>>;
}

interface FilterOptions {
  category?: string;
  keyword?: string;
}

interface Review {
  id: string;
  productId: string;
  author: string;
  rating: number; // 1-5
  comment: string;
  date: string;
}

interface Coupon {
  code: string;
  type: "percent" | "flat";
  value: number;
  minOrderValue?: number;
  expiry?: string; // YYYY-MM-DD
}

interface CouponResult {
  valid: boolean;
  message: string;
  discountAmount: number;
}

type ToastType = "success" | "error" | "info" | "warning";

interface UserAccount {
  email: string;
  passwordHash: string;
  name: string;
}

interface Order {
  id: string;
  items: CartItem[];
  total: number;
  placedAt: string;
  status: "pending" | "confirmed" | "delivered" | "cancelled";
}

declare global {
  interface Window {
    PRODUCTS?: Product[];
  }
}

/* --------------------------------------------------------------------------
   1. CART MODULE
   -------------------------------------------------------------------------- */
const Cart = (() => {
  const STORAGE_KEY = "site_cart_items";

  function getItems(): CartItem[] {
    const raw = localStorage.getItem(STORAGE_KEY);
    try {
      return raw ? (JSON.parse(raw) as CartItem[]) : [];
    } catch (err) {
      console.error("Cart parse error:", err);
      return [];
    }
  }

  function saveItems(items: CartItem[]): void {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(items));
    updateCartBadge();
  }

  function addItem(product: Product, quantity: number = 1): CartItem[] {
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

  function removeItem(productId: string): CartItem[] {
    const items = getItems().filter((item) => item.id !== productId);
    saveItems(items);
    return items;
  }

  function updateQuantity(productId: string, quantity: number): CartItem[] {
    const items = getItems();
    const item = items.find((i) => i.id === productId);
    if (item) {
      item.quantity = Math.max(1, quantity);
    }
    saveItems(items);
    return items;
  }

  function clearCart(): void {
    saveItems([]);
  }

  function getTotal(): number {
    return getItems().reduce((sum, item) => sum + item.price * item.quantity, 0);
  }

  function getItemCount(): number {
    return getItems().reduce((sum, item) => sum + item.quantity, 0);
  }

  function updateCartBadge(): void {
    const badge = document.querySelector<HTMLElement>("[data-cart-count]");
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
function filterProducts(products: Product[], { category = "all", keyword = "" }: FilterOptions = {}): Product[] {
  const normalizedKeyword = keyword.trim().toLowerCase();

  return products.filter((product) => {
    const matchesCategory =
      category === "all" || product.category.toLowerCase() === category.toLowerCase();

    const matchesKeyword =
      normalizedKeyword === "" || product.name.toLowerCase().includes(normalizedKeyword);

    return matchesCategory && matchesKeyword;
  });
}

function sortProducts(products: Product[], sortBy: "price-asc" | "price-desc" | "name-asc" | "name-desc"): Product[] {
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

function renderProductGrid(products: Product[], containerSelector: string): void {
  const container = document.querySelector<HTMLElement>(containerSelector);
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
  function isEmpty(value: string): boolean {
    return !value || value.trim().length === 0;
  }

  function isValidEmail(email: string): boolean {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
  }

  function isValidPhone(phone: string): boolean {
    return /^[0-9]{10}$/.test(phone.trim());
  }

  function isFutureDate(dateStr: string): boolean {
    const inputDate = new Date(dateStr);
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    return inputDate >= today;
  }

  function isPositiveInteger(value: string): boolean {
    const num = Number(value);
    return Number.isInteger(num) && num > 0;
  }

  function validateBookingForm(formData: BookingFormData): ValidationResult {
    const errors: ValidationResult["errors"] = {};

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

  function showErrors(errors: ValidationResult["errors"]): void {
    document.querySelectorAll<HTMLElement>(".error-message").forEach((el) => (el.textContent = ""));
    (Object.entries(errors) as [keyof BookingFormData, string][]).forEach(([field, message]) => {
      const el = document.querySelector<HTMLElement>(`[data-error-for="${field}"]`);
      if (el) el.textContent = message;
    });
  }

  return { validateBookingForm, showErrors, isValidEmail, isValidPhone, isFutureDate };
})();

/* --------------------------------------------------------------------------
   4. TICKET / VISITOR COUNTER WIDGET (Museum & Zoo ticketing)
   -------------------------------------------------------------------------- */
function initVisitorCounter(counterSelector: string, pricePerTicket: number): void {
  const container = document.querySelector<HTMLElement>(counterSelector);
  if (!container) return;

  const decrementBtn = container.querySelector<HTMLButtonElement>("[data-action='decrement']");
  const incrementBtn = container.querySelector<HTMLButtonElement>("[data-action='increment']");
  const countDisplay = container.querySelector<HTMLElement>("[data-visitor-count]");
  const totalDisplay = container.querySelector<HTMLElement>("[data-total-price]");

  let count = 1;

  function render(): void {
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
function debounce<T extends (...args: any[]) => void>(fn: T, delay: number = 300): (...args: Parameters<T>) => void {
  let timeoutId: ReturnType<typeof setTimeout>;
  return function debounced(this: unknown, ...args: Parameters<T>): void {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => fn.apply(this, args), delay);
  };
}

function throttle<T extends (...args: any[]) => void>(fn: T, limit: number = 300): (...args: Parameters<T>) => void {
  let waiting = false;
  return function throttled(this: unknown, ...args: Parameters<T>): void {
    if (!waiting) {
      fn.apply(this, args);
      waiting = true;
      setTimeout(() => (waiting = false), limit);
    }
  };
}

function delegateEvent(
  parentSelector: string,
  eventType: string,
  childSelector: string,
  handler: (event: Event, target: HTMLElement) => void
): void {
  const parent = document.querySelector<HTMLElement>(parentSelector);
  if (!parent) return;

  parent.addEventListener(eventType, (event: Event) => {
    const eventTarget = event.target as HTMLElement;
    const target = eventTarget.closest<HTMLElement>(childSelector);
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

  function getItems(): Product[] {
    const raw = localStorage.getItem(STORAGE_KEY);
    try {
      return raw ? (JSON.parse(raw) as Product[]) : [];
    } catch {
      return [];
    }
  }

  function save(items: Product[]): void {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(items));
    updateBadge();
  }

  function toggle(product: Product): boolean {
    const items = getItems();
    const index = items.findIndex((p) => p.id === product.id);
    let added: boolean;

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

  function isWishlisted(productId: string): boolean {
    return getItems().some((p) => p.id === productId);
  }

  function updateBadge(): void {
    const badge = document.querySelector<HTMLElement>("[data-wishlist-count]");
    if (badge) badge.textContent = String(getItems().length);
  }

  return { getItems, toggle, isWishlisted, updateBadge };
})();

/* --------------------------------------------------------------------------
   7. PAGINATION MODULE
   -------------------------------------------------------------------------- */
class Paginator<T> {
  private items: T[];
  private pageSize: number;
  private currentPage: number = 1;

  constructor(items: T[], pageSize: number = 6) {
    this.items = items;
    this.pageSize = pageSize;
  }

  get totalPages(): number {
    return Math.max(1, Math.ceil(this.items.length / this.pageSize));
  }

  getPage(page: number): T[] {
    this.currentPage = Math.min(Math.max(1, page), this.totalPages);
    const start = (this.currentPage - 1) * this.pageSize;
    return this.items.slice(start, start + this.pageSize);
  }

  next(): T[] {
    return this.getPage(this.currentPage + 1);
  }

  prev(): T[] {
    return this.getPage(this.currentPage - 1);
  }

  getCurrentPage(): number {
    return this.currentPage;
  }

  updateItems(items: T[]): void {
    this.items = items;
    this.currentPage = 1;
  }

  renderControls(containerSelector: string, onPageChange: (page: number) => void): void {
    const container = document.querySelector<HTMLElement>(containerSelector);
    if (!container) return;

    const buttons: string[] = [];
    for (let i = 1; i <= this.totalPages; i++) {
      buttons.push(
        `<button class="btn btn-sm ${i === this.currentPage ? "btn-primary" : "btn-outline-secondary"} page-btn" data-page="${i}">${i}</button>`
      );
    }
    container.innerHTML = buttons.join(" ");

    container.querySelectorAll<HTMLButtonElement>(".page-btn").forEach((btn) => {
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

  function getAll(): Review[] {
    const raw = localStorage.getItem(STORAGE_KEY);
    try {
      return raw ? (JSON.parse(raw) as Review[]) : [];
    } catch {
      return [];
    }
  }

  function getForProduct(productId: string): Review[] {
    return getAll().filter((r) => r.productId === productId);
  }

  function addReview(review: Omit<Review, "id" | "date">): Review {
    const newReview: Review = {
      ...review,
      id: `rev_${Date.now()}_${Math.floor(Math.random() * 1000)}`,
      date: new Date().toISOString().split("T")[0],
    };
    const all = getAll();
    all.push(newReview);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(all));
    return newReview;
  }

  function getAverageRating(productId: string): number {
    const reviews = getForProduct(productId);
    if (reviews.length === 0) return 0;
    const total = reviews.reduce((sum, r) => sum + r.rating, 0);
    return Math.round((total / reviews.length) * 10) / 10;
  }

  function renderStars(rating: number): string {
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
  const AVAILABLE_COUPONS: Coupon[] = [
    { code: "SAVE10", type: "percent", value: 10, minOrderValue: 500 },
    { code: "FLAT50", type: "flat", value: 50, minOrderValue: 300 },
    { code: "WELCOME20", type: "percent", value: 20, minOrderValue: 1000, expiry: "2026-12-31" },
  ];

  function isExpired(coupon: Coupon): boolean {
    if (!coupon.expiry) return false;
    return new Date(coupon.expiry) < new Date();
  }

  function applyCoupon(code: string, orderTotal: number): CouponResult {
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
  let container: HTMLElement | null = null;

  function ensureContainer(): HTMLElement {
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

  function colorFor(type: ToastType): string {
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

  function show(message: string, type: ToastType = "info", duration: number = 3000): void {
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
  function open(modalSelector: string): void {
    const modal = document.querySelector<HTMLElement>(modalSelector);
    if (!modal) return;
    modal.classList.add("show");
    modal.style.display = "block";
    document.body.classList.add("modal-open");
  }

  function close(modalSelector: string): void {
    const modal = document.querySelector<HTMLElement>(modalSelector);
    if (!modal) return;
    modal.classList.remove("show");
    modal.style.display = "none";
    document.body.classList.remove("modal-open");
  }

  function bindCloseTriggers(modalSelector: string, closeTriggerSelector: string): void {
    document.querySelectorAll<HTMLElement>(closeTriggerSelector).forEach((trigger) => {
      trigger.addEventListener("click", () => close(modalSelector));
    });
  }

  return { open, close, bindCloseTriggers };
})();

/* --------------------------------------------------------------------------
   12. LAZY-LOADING IMAGES (IntersectionObserver)
   -------------------------------------------------------------------------- */
function initLazyLoad(imageSelector: string): void {
  const images = document.querySelectorAll<HTMLImageElement>(imageSelector);
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
          const img = entry.target as HTMLImageElement;
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

  function simpleHash(value: string): string {
    let hash = 0;
    for (let i = 0; i < value.length; i++) {
      hash = (hash << 5) - hash + value.charCodeAt(i);
      hash |= 0;
    }
    return String(hash);
  }

  function getUsers(): UserAccount[] {
    const raw = localStorage.getItem(USERS_KEY);
    try {
      return raw ? (JSON.parse(raw) as UserAccount[]) : [];
    } catch {
      return [];
    }
  }

  function register(name: string, email: string, password: string): { success: boolean; message: string } {
    const users = getUsers();
    if (users.some((u) => u.email.toLowerCase() === email.toLowerCase())) {
      return { success: false, message: "An account with this email already exists." };
    }
    users.push({ name, email, passwordHash: simpleHash(password) });
    localStorage.setItem(USERS_KEY, JSON.stringify(users));
    return { success: true, message: "Registration successful. Please log in." };
  }

  function login(email: string, password: string): { success: boolean; message: string } {
    const users = getUsers();
    const user = users.find((u) => u.email.toLowerCase() === email.toLowerCase());

    if (!user || user.passwordHash !== simpleHash(password)) {
      return { success: false, message: "Invalid email or password." };
    }

    localStorage.setItem(SESSION_KEY, JSON.stringify({ email: user.email, name: user.name }));
    return { success: true, message: `Welcome back, ${user.name}!` };
  }

  function logout(): void {
    localStorage.removeItem(SESSION_KEY);
  }

  function getCurrentUser(): { email: string; name: string } | null {
    const raw = localStorage.getItem(SESSION_KEY);
    try {
      return raw ? JSON.parse(raw) : null;
    } catch {
      return null;
    }
  }

  function isLoggedIn(): boolean {
    return getCurrentUser() !== null;
  }

  return { register, login, logout, getCurrentUser, isLoggedIn };
})();

/* --------------------------------------------------------------------------
   14. ORDER HISTORY
   -------------------------------------------------------------------------- */
const OrderHistory = (() => {
  const STORAGE_KEY = "site_order_history";

  function getAll(): Order[] {
    const raw = localStorage.getItem(STORAGE_KEY);
    try {
      return raw ? (JSON.parse(raw) as Order[]) : [];
    } catch {
      return [];
    }
  }

  function placeOrder(items: CartItem[]): Order {
    const order: Order = {
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

  function updateStatus(orderId: string, status: Order["status"]): void {
    const all = getAll();
    const order = all.find((o) => o.id === orderId);
    if (order) order.status = status;
    localStorage.setItem(STORAGE_KEY, JSON.stringify(all));
  }

  function renderHistory(containerSelector: string): void {
    const container = document.querySelector<HTMLElement>(containerSelector);
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

  const searchInput = document.querySelector<HTMLInputElement>("[data-search-input]");
  const categoryFilter = document.querySelector<HTMLSelectElement>("[data-category-filter]");
  const sortSelect = document.querySelector<HTMLSelectElement>("[data-sort-select]");

  function refreshGrid(): void {
    if (!window.PRODUCTS) return;
    let result = filterProducts(window.PRODUCTS, {
      keyword: searchInput?.value ?? "",
      category: categoryFilter?.value ?? "all",
    });
    if (sortSelect?.value) {
      result = sortProducts(result, sortSelect.value as any);
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
  const couponForm = document.querySelector<HTMLFormElement>("#couponForm");
  if (couponForm) {
    couponForm.addEventListener("submit", (e: SubmitEvent) => {
      e.preventDefault();
      const input = couponForm.querySelector<HTMLInputElement>("[name='coupon']");
      if (!input) return;
      const result = CouponEngine.applyCoupon(input.value, Cart.getTotal());
      Toast.show(result.message, result.valid ? "success" : "error");
    });
  }

  // Booking form submission
  const bookingForm = document.querySelector<HTMLFormElement>("#bookingForm");
  if (bookingForm) {
    bookingForm.addEventListener("submit", (e: SubmitEvent) => {
      e.preventDefault();
      const formData = Object.fromEntries(new FormData(bookingForm).entries()) as unknown as BookingFormData;
      const { isValid, errors } = Validator.validateBookingForm(formData);

      Validator.showErrors(errors);

      if (isValid) {
        Toast.show("Booking confirmed! Thank you.", "success");
        bookingForm.reset();
      }
    });
  }

  // Review form
  const reviewForm = document.querySelector<HTMLFormElement>("#reviewForm");
  if (reviewForm) {
    reviewForm.addEventListener("submit", (e: SubmitEvent) => {
      e.preventDefault();
      const data = Object.fromEntries(new FormData(reviewForm).entries()) as Record<string, string>;
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
  const loginForm = document.querySelector<HTMLFormElement>("#loginForm");
  loginForm?.addEventListener("submit", (e: SubmitEvent) => {
    e.preventDefault();
    const data = Object.fromEntries(new FormData(loginForm).entries()) as Record<string, string>;
    const result = Auth.login(data.email, data.password);
    Toast.show(result.message, result.success ? "success" : "error");
  });

  const registerForm = document.querySelector<HTMLFormElement>("#registerForm");
  registerForm?.addEventListener("submit", (e: SubmitEvent) => {
    e.preventDefault();
    const data = Object.fromEntries(new FormData(registerForm).entries()) as Record<string, string>;
    const result = Auth.register(data.name, data.email, data.password);
    Toast.show(result.message, result.success ? "success" : "error");
  });

  // Checkout button -> place order
  const checkoutBtn = document.querySelector<HTMLButtonElement>("[data-checkout-btn]");
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
  document.querySelectorAll<HTMLElement>("[data-modal-open]").forEach((trigger) => {
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

export {};
