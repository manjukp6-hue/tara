"""
python/tara_setup/ui.py

Graphical User Interface (Tkinter) for TARA Setup.
100% LOCAL ONLY Creator Setup and Device Identity Management.
Provides:
 1. Native Windows dark-mode UI.
 2. Real-time local authority state inspection (CREATOR_SETUP_REQUIRED / ACTIVE / AUTHORITY_LOCKED).
 3. Interactive browser Google login with local loopback callback (OpenID Connect + PKCE).
 4. Explicit first-time confirmation check before generating creator trust root.
 5. Secure passphrase entry and Scrypt + AES-256-GCM + DPAPI key protection.
 6. Emergency Recovery Code modal dialog with one-click copy.
 7. Any-PC secondary device registration flow without creator private key copying.
 8. Device management and instant revocation controls using human-readable device names.
 9. Secure local Creator Console handoff launcher.
"""

import os
import sys
import json
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional, Dict, Any, List

from .client import (
    SetupClient,
    CANONICAL_CREATOR_ID,
    CANONICAL_DISPLAY_NAME
)


class SetupGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("TARA Setup — Creator Trust Root")
        self.root.geometry("680x740")
        self.root.minsize(620, 660)

        # Apply dark theme palette
        self.bg_main = "#0b0f19"
        self.bg_card = "#161f36"
        self.bg_input = "#0f172a"
        self.fg_main = "#f8fafc"
        self.fg_muted = "#94a3b8"
        self.color_cyan = "#06b6d4"
        self.color_emerald = "#10b981"
        self.color_amber = "#f59e0b"
        self.color_rose = "#f43f5e"

        self.root.configure(bg=self.bg_main)
        self.client = SetupClient()

        self.google_token: Optional[str] = None
        self.google_nonce: Optional[str] = None
        self.google_email: Optional[str] = None
        self.recovery_code: Optional[str] = None
        self._devices_cache: List[Dict[str, Any]] = []

        self._build_styles()
        self._build_ui()
        self._refresh_authority_state_async()

    def _build_styles(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure(".", background=self.bg_main, foreground=self.fg_main)
        style.configure("TFrame", background=self.bg_main)
        style.configure("Card.TFrame", background=self.bg_card, relief="flat")
        style.configure("TLabel", background=self.bg_main, foreground=self.fg_main, font=("Segoe UI", 10))
        style.configure("Card.TLabel", background=self.bg_card, foreground=self.fg_main, font=("Segoe UI", 10))
        style.configure("Muted.TLabel", background=self.bg_card, foreground=self.fg_muted, font=("Segoe UI", 9))
        style.configure("Title.TLabel", background=self.bg_main, foreground=self.color_cyan, font=("Segoe UI Semibold", 18))
        style.configure("Header.TLabel", background=self.bg_card, foreground=self.fg_main, font=("Segoe UI Semibold", 12))
        style.configure("TEntry", fieldbackground=self.bg_input, foreground=self.fg_main, insertcolor="#fff")
        style.configure("TCheckbutton", background=self.bg_card, foreground=self.fg_main, font=("Segoe UI", 10))

    def _build_ui(self):
        # 1. Header Frame
        header = ttk.Frame(self.root, padding="16 16 16 8")
        header.pack(fill="x")

        ttk.Label(header, text="TARA Setup", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Creator Root Trust Initialization & Local Device Identity",
            style="TLabel",
            foreground=self.fg_muted
        ).pack(anchor="w")

        # 2. Local Authority State Card
        conn_frame = ttk.Frame(self.root, style="Card.TFrame", padding="12")
        conn_frame.pack(fill="x", padx=16, pady=8)

        conn_top = ttk.Frame(conn_frame, style="Card.TFrame")
        conn_top.pack(fill="x")

        ttk.Label(conn_top, text="Local Authority State:", style="Card.TLabel").pack(side="left")

        self.status_badge = tk.Label(
            conn_top,
            text="CHECKING...",
            bg="#1e293b",
            fg=self.fg_muted,
            font=("Segoe UI Bold", 9),
            padx=10,
            pady=3
        )
        self.status_badge.pack(side="left", padx=10)

        self.btn_refresh = tk.Button(
            conn_top,
            text="Refresh State",
            command=self._refresh_authority_state_async,
            bg="#232f48",
            fg="white",
            relief="flat",
            padx=12,
            pady=3,
            cursor="hand2"
        )
        self.btn_refresh.pack(side="right")

        # 3. Main Content Notebook (Tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=16, pady=8)

        self.tab_setup = ttk.Frame(self.notebook, padding="12")
        self.tab_pc2 = ttk.Frame(self.notebook, padding="12")
        self.tab_devices = ttk.Frame(self.notebook, padding="12")

        self.notebook.add(self.tab_setup, text=" First-Time Setup ")
        self.notebook.add(self.tab_pc2, text=" Any-PC Device Registration ")
        self.notebook.add(self.tab_devices, text=" Device Management ")

        self._build_setup_tab()
        self._build_pc2_tab()
        self._build_devices_tab()

        # 4. Footer log/status label
        self.footer_label = tk.Label(
            self.root,
            text="Ready. Creator trust root required before production use.",
            bg=self.bg_main,
            fg=self.fg_muted,
            font=("Segoe UI", 9),
            anchor="w",
            padx=16,
            pady=8
        )
        self.footer_label.pack(fill="x", side="bottom")

    # -------------------------------------------------------------------------
    # TAB 1: FIRST-TIME ROOT CREATOR SETUP
    # -------------------------------------------------------------------------
    def _build_setup_tab(self):
        container = ttk.Frame(self.tab_setup, style="Card.TFrame", padding="14")
        container.pack(fill="both", expand=True)

        ttk.Label(
            container,
            text="1. Authenticate Creator Google Account",
            style="Header.TLabel"
        ).pack(anchor="w")

        google_row = ttk.Frame(container, style="Card.TFrame")
        google_row.pack(fill="x", pady=6)

        self.btn_google = tk.Button(
            google_row,
            text="🔐 Sign In With Google (System Browser)",
            command=self._launch_google_login_thread,
            bg="#1d4ed8",
            fg="white",
            font=("Segoe UI Semibold", 10),
            relief="flat",
            padx=14,
            pady=6,
            cursor="hand2"
        )
        self.btn_google.pack(side="left")

        self.google_status_lbl = tk.Label(
            container,
            text="Not authenticated (Google sign-in required to establish Creator Authority)",
            bg=self.bg_card,
            fg=self.fg_muted,
            font=("Segoe UI", 9)
        )
        self.google_status_lbl.pack(anchor="w", pady=(0, 10))

        ttk.Separator(container, orient="horizontal").pack(fill="x", pady=6)

        # 2. Explicit confirmation
        ttk.Label(container, text="2. Explicit Root Creator Confirmation", style="Header.TLabel").pack(anchor="w")
        self.confirm_var = tk.BooleanVar(value=False)
        self.chk_confirm = tk.Checkbutton(
            container,
            text="I explicitly confirm this is the initial Root Creator authority setup for this installation.",
            variable=self.confirm_var,
            bg=self.bg_card,
            fg=self.fg_main,
            selectcolor=self.bg_input,
            activebackground=self.bg_card,
            activeforeground=self.fg_main,
            font=("Segoe UI", 10),
            wraplength=560,
            justify="left"
        )
        self.chk_confirm.pack(anchor="w", pady=6)

        ttk.Separator(container, orient="horizontal").pack(fill="x", pady=6)

        # 3. Passphrase fields
        ttk.Label(container, text="3. Root Keystore Protection (Scrypt + AES-256-GCM + DPAPI)", style="Header.TLabel").pack(anchor="w")
        pass_grid = ttk.Frame(container, style="Card.TFrame")
        pass_grid.pack(fill="x", pady=4)

        ttk.Label(pass_grid, text="Master Passphrase:", style="Card.TLabel").grid(row=0, column=0, sticky="w", pady=3)
        self.pass_entry = tk.Entry(pass_grid, show="●", bg=self.bg_input, fg=self.fg_main, insertbackground="white", bd=1, relief="solid", width=30)
        self.pass_entry.grid(row=0, column=1, sticky="w", padx=8, pady=3)

        ttk.Label(pass_grid, text="Confirm Passphrase:", style="Card.TLabel").grid(row=1, column=0, sticky="w", pady=3)
        self.pass_conf_entry = tk.Entry(pass_grid, show="●", bg=self.bg_input, fg=self.fg_main, insertbackground="white", bd=1, relief="solid", width=30)
        self.pass_conf_entry.grid(row=1, column=1, sticky="w", padx=8, pady=3)

        ttk.Separator(container, orient="horizontal").pack(fill="x", pady=6)

        # 4. Device Name
        ttk.Label(container, text="4. PC Device Name", style="Header.TLabel").pack(anchor="w")
        meta_grid = ttk.Frame(container, style="Card.TFrame")
        meta_grid.pack(fill="x", pady=4)

        ttk.Label(meta_grid, text="This PC Name:", style="Card.TLabel").grid(row=0, column=0, sticky="w", pady=3)
        self.dev_name_entry = tk.Entry(meta_grid, bg=self.bg_input, fg=self.fg_main, insertbackground="white", bd=1, relief="solid", width=30)
        import platform
        self.dev_name_entry.insert(0, platform.node() or "Primary Windows PC")
        self.dev_name_entry.grid(row=0, column=1, sticky="w", padx=8, pady=3)

        # Action Buttons
        btn_box = ttk.Frame(container, style="Card.TFrame")
        btn_box.pack(fill="x", pady=(14, 4))

        self.btn_execute_setup = tk.Button(
            btn_box,
            text="🚀 Initialize Creator Trust Root",
            command=self._execute_setup_thread,
            bg=self.color_emerald,
            fg="#052e16",
            font=("Segoe UI Bold", 11),
            relief="flat",
            padx=16,
            pady=8,
            cursor="hand2"
        )
        self.btn_execute_setup.pack(side="left")

        self.btn_open_console = tk.Button(
            btn_box,
            text="🌐 Open Creator Console",
            command=self._launch_console,
            bg="#2563eb",
            fg="white",
            font=("Segoe UI Semibold", 10),
            relief="flat",
            padx=12,
            pady=8,
            cursor="hand2",
            state="disabled"
        )
        self.btn_open_console.pack(side="left", padx=12)

    # -------------------------------------------------------------------------
    # TAB 2: ANY-PC DEVICE REGISTRATION (FOR PC 2+)
    # -------------------------------------------------------------------------
    def _build_pc2_tab(self):
        container = ttk.Frame(self.tab_pc2, style="Card.TFrame", padding="16")
        container.pack(fill="both", expand=True)

        ttk.Label(
            container,
            text="Any-PC Creator Device Registration",
            style="Header.TLabel"
        ).pack(anchor="w")

        ttk.Label(
            container,
            text=(
                "When local TARA authority is ACTIVE, you can securely access the Creator Console "
                "from any secondary Windows PC without copying the root creator private key.\n"
                "A local machine keypair is generated and bound to this PC via Windows DPAPI."
            ),
            style="Muted.TLabel",
            wraplength=560,
            justify="left"
        ).pack(anchor="w", pady=(4, 12))

        # Google login for PC 2
        ttk.Label(container, text="1. Verify Creator Identity (Google)", style="Header.TLabel").pack(anchor="w")
        pc2_g_row = ttk.Frame(container, style="Card.TFrame")
        pc2_g_row.pack(fill="x", pady=6)

        self.btn_pc2_google = tk.Button(
            pc2_g_row,
            text="🔐 Authenticate with Google",
            command=self._launch_google_login_thread,
            bg="#1d4ed8",
            fg="white",
            font=("Segoe UI Semibold", 10),
            relief="flat",
            padx=14,
            pady=6,
            cursor="hand2"
        )
        self.btn_pc2_google.pack(side="left")

        self.pc2_google_status = tk.Label(
            container,
            text="Not authenticated",
            bg=self.bg_card,
            fg=self.fg_muted,
            font=("Segoe UI", 9)
        )
        self.pc2_google_status.pack(anchor="w", pady=(0, 10))

        ttk.Label(container, text="2. Device Information", style="Header.TLabel").pack(anchor="w")
        pc2_dev_row = ttk.Frame(container, style="Card.TFrame")
        pc2_dev_row.pack(fill="x", pady=6)

        ttk.Label(pc2_dev_row, text="Secondary PC Name:", style="Card.TLabel").pack(side="left")
        import platform
        self.pc2_name_entry = tk.Entry(pc2_dev_row, bg=self.bg_input, fg=self.fg_main, insertbackground="white", bd=1, relief="solid", width=30)
        self.pc2_name_entry.insert(0, f"Secondary PC ({platform.node()})")
        self.pc2_name_entry.pack(side="left", padx=8)

        ttk.Separator(container, orient="horizontal").pack(fill="x", pady=12)

        self.btn_register_pc2 = tk.Button(
            container,
            text="💻 Register & Authorize This PC",
            command=self._execute_pc2_register_thread,
            bg=self.color_cyan,
            fg="#083344",
            font=("Segoe UI Bold", 11),
            relief="flat",
            padx=16,
            pady=8,
            cursor="hand2"
        )
        self.btn_register_pc2.pack(anchor="w", pady=6)

    # -------------------------------------------------------------------------
    # TAB 3: DEVICE MANAGEMENT
    # -------------------------------------------------------------------------
    def _build_devices_tab(self):
        container = ttk.Frame(self.tab_devices, style="Card.TFrame", padding="16")
        container.pack(fill="both", expand=True)

        ttk.Label(container, text="Registered Devices & Lost-Device Protection", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            container,
            text="Revoked devices instantly lose all creator capabilities on the local TARA installation.",
            style="Muted.TLabel"
        ).pack(anchor="w", pady=(2, 8))

        # Device listbox
        list_frame = ttk.Frame(container, style="Card.TFrame")
        list_frame.pack(fill="both", expand=True, pady=6)

        self.device_listbox = tk.Listbox(
            list_frame,
            bg=self.bg_input,
            fg=self.fg_main,
            selectbackground="#2563eb",
            selectforeground="white",
            font=("Segoe UI", 10),
            relief="solid",
            bd=1
        )
        self.device_listbox.pack(side="left", fill="both", expand=True)

        scroll = tk.Scrollbar(list_frame, orient="vertical", command=self.device_listbox.yview)
        scroll.pack(side="right", fill="y")
        self.device_listbox.config(yscrollcommand=scroll.set)

        dev_btn_box = ttk.Frame(container, style="Card.TFrame")
        dev_btn_box.pack(fill="x", pady=8)

        self.btn_list_devs = tk.Button(
            dev_btn_box,
            text="🔄 Refresh Devices",
            command=self._refresh_device_list_async,
            bg="#334155",
            fg="white",
            relief="flat",
            padx=10,
            pady=6,
            cursor="hand2"
        )
        self.btn_list_devs.pack(side="left")

        self.btn_revoke_dev = tk.Button(
            dev_btn_box,
            text="🚫 Revoke Selected Device",
            command=self._revoke_selected_device,
            bg=self.color_rose,
            fg="white",
            relief="flat",
            padx=10,
            pady=6,
            cursor="hand2"
        )
        self.btn_revoke_dev.pack(side="left", padx=10)

    # -------------------------------------------------------------------------
    # LOCAL AUTHORITY STATE INSPECTION
    # -------------------------------------------------------------------------
    def _refresh_authority_state_async(self):
        def worker():
            self.status_badge.config(text="CHECKING...", bg="#334155", fg="white")
            state_info = self.client.get_local_authority_state()
            self.root.after(0, lambda: self._apply_authority_state(state_info))

        threading.Thread(target=worker, daemon=True).start()

    def _apply_authority_state(self, state_info: Dict[str, Any]):
        c_status = state_info.get("creator_status", "UNKNOWN")

        if c_status == "CREATOR_SETUP_REQUIRED":
            self.status_badge.config(text="● CREATOR_SETUP_REQUIRED", bg="#854d0e", fg="#fef08a")
            self.footer_label.config(
                text="Local authority is uninitialized. Complete Google Authentication to establish Root Creator.",
                fg=self.color_amber
            )
            self.btn_execute_setup.config(state="normal")
            self.notebook.select(self.tab_setup)
        elif c_status == "ACTIVE":
            self.status_badge.config(text="● ACTIVE (INITIALIZED)", bg="#065f46", fg="#a7f3d0")
            self.footer_label.config(
                text="Creator Authority is ACTIVE. Use 'Any-PC Device Registration' to authorize this computer.",
                fg=self.color_emerald
            )
            self.btn_execute_setup.config(state="disabled")
            self.btn_open_console.config(state="normal")
            self.notebook.select(self.tab_pc2)
        elif c_status == "AUTHORITY_LOCKED":
            self.status_badge.config(text="● AUTHORITY_LOCKED", bg=self.color_rose, fg="white")
            self.footer_label.config(
                text="Creator authority is LOCKED due to tamper detection or missing cryptographic seal.",
                fg=self.color_rose
            )
            self.btn_execute_setup.config(state="disabled")
        else:
            self.status_badge.config(text=f"● {c_status}", bg="#334155", fg="white")

    # -------------------------------------------------------------------------
    # GOOGLE AUTH ACTIONS (OPENID CONNECT + PKCE)
    # -------------------------------------------------------------------------
    def _launch_google_login_thread(self):
        self.footer_label.config(text="Opening browser for Google login... Complete login in browser.", fg=self.color_cyan)

        def worker():
            try:
                flow = self.client.start_google_login_flow(timeout_seconds=90.0, open_browser=True)
                flow["thread"].join(timeout=95.0)
                res = flow["result_holder"]
                if res.get("token"):
                    nonce = flow.get("nonce")
                    self.root.after(0, lambda: self._on_google_token_received(res["token"], nonce))
                else:
                    err = res.get("error", "No token received.")
                    self.root.after(0, lambda: self._on_google_login_error(err))
            except Exception as e:
                self.root.after(0, lambda: self._on_google_login_error(str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_google_token_received(self, token_str: str, expected_nonce: Optional[str] = None):
        verif = self.client.verify_google_token_string(token_str, expected_nonce=expected_nonce)
        if verif.get("valid"):
            self.google_token = token_str
            self.google_nonce = expected_nonce
            self.google_email = verif["email"]
            # Show generic verified status, avoiding unnecessary exposure of raw email
            msg = "✓ Google Identity Cryptographically Verified"
            self.google_status_lbl.config(text=msg, fg=self.color_emerald)
            self.pc2_google_status.config(text=msg, fg=self.color_emerald)
            self.footer_label.config(text="Google identity verified. Proceed with confirmation and passphrase entry.", fg=self.color_emerald)
        else:
            self.google_token = None
            self.google_nonce = None
            self.google_email = None
            err_msg = f"Verification Failed: {verif.get('error')}"
            self.google_status_lbl.config(text=err_msg, fg=self.color_rose)
            self.pc2_google_status.config(text=err_msg, fg=self.color_rose)
            messagebox.showerror("Google Verification Error", verif.get("error", "Invalid ID token."))

    def _on_google_login_error(self, err_msg: str):
        self.footer_label.config(text=f"Google login notice: {err_msg}", fg=self.color_amber)

    # -------------------------------------------------------------------------
    # FIRST-TIME SETUP EXECUTION
    # -------------------------------------------------------------------------
    def _execute_setup_thread(self):
        if not self.google_token:
            messagebox.showerror("Prerequisite Missing", "You must first authenticate with your Google account.")
            return

        if not self.confirm_var.get():
            messagebox.showerror("Confirmation Required", "You must check the confirmation checkbox declaring this is the initial Root Creator setup.")
            return

        p1 = self.pass_entry.get()
        p2 = self.pass_conf_entry.get()
        if not p1 or len(p1) < 8:
            messagebox.showerror("Passphrase Error", "Master passphrase must be at least 8 characters.")
            return
        if p1 != p2:
            messagebox.showerror("Passphrase Error", "Master passphrase and confirmation do not match.")
            return

        dev_name = self.dev_name_entry.get().strip() or "Primary Windows PC"

        self.btn_execute_setup.config(state="disabled", text="Initializing Trust Root...")
        self.footer_label.config(text="Generating cryptographic root keys and sealing authority...", fg=self.color_cyan)

        # Retain token for worker invocation
        tok = self.google_token
        nonce = self.google_nonce

        def worker():
            try:
                res = self.client.perform_first_time_setup(
                    master_passphrase=p1,
                    confirm_passphrase=p2,
                    google_id_token=tok,
                    confirm_identity=True,
                    device_name=dev_name,
                    expected_nonce=nonce
                )
                self.root.after(0, lambda: self._on_setup_success(res))
            except Exception as e:
                self.root.after(0, lambda: self._on_setup_failed(str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_setup_success(self, res: Dict[str, Any]):
        # Clear sensitive temporary token and passphrases from memory immediately
        self.google_token = None
        self.google_nonce = None
        self.google_email = None
        self.pass_entry.delete(0, tk.END)
        self.pass_conf_entry.delete(0, tk.END)

        self.btn_execute_setup.config(text="✓ Setup Complete", bg="#065f46", fg="white")
        self.btn_open_console.config(state="normal")
        self.recovery_code = res.get("recovery_code")
        self._refresh_authority_state_async()

        # Display recovery code modal
        self._show_recovery_modal(self.recovery_code)

    def _on_setup_failed(self, err_msg: str):
        self.btn_execute_setup.config(state="normal", text="🚀 Initialize Creator Trust Root")
        self.footer_label.config(text=f"Setup failed: {err_msg}", fg=self.color_rose)
        messagebox.showerror("Setup Failure", f"Failed to initialize creator authority:\n\n{err_msg}")

    def _show_recovery_modal(self, code: Optional[str]):
        if not code:
            messagebox.showinfo("Success", "Root Creator trust root initialized successfully!")
            return

        modal = tk.Toplevel(self.root)
        modal.title("CRITICAL: Save Emergency Recovery Code")
        modal.geometry("580x360")
        modal.configure(bg=self.bg_main)
        modal.transient(self.root)
        modal.grab_set()

        ttk.Label(
            modal,
            text="⚠️ Emergency Recovery Code Generated",
            style="Title.TLabel"
        ).pack(anchor="w", padx=20, pady=(20, 4))

        ttk.Label(
            modal,
            text=(
                "Store this code in an offline, physically secure location. "
                "It is strictly required if you ever lose your creator device or master password.\n"
                "This code will NEVER be shown again."
            ),
            style="TLabel",
            foreground=self.color_amber,
            wraplength=540,
            justify="left"
        ).pack(anchor="w", padx=20, pady=(0, 12))

        code_box = tk.Entry(
            modal,
            font=("Consolas", 14, "bold"),
            bg=self.bg_input,
            fg=self.color_emerald,
            justify="center",
            relief="solid",
            bd=1
        )
        code_box.insert(0, code)
        code_box.configure(state="readonly")
        code_box.pack(fill="x", padx=20, pady=8)

        def copy_code():
            self.root.clipboard_clear()
            self.root.clipboard_append(code)
            messagebox.showinfo("Copied", "Recovery code copied to clipboard!", parent=modal)

        btn_row = ttk.Frame(modal)
        btn_row.pack(fill="x", padx=20, pady=16)

        tk.Button(
            btn_row,
            text="📋 Copy Recovery Code",
            command=copy_code,
            bg=self.color_cyan,
            fg="#083344",
            font=("Segoe UI Semibold", 10),
            relief="flat",
            padx=14,
            pady=6,
            cursor="hand2"
        ).pack(side="left")

        tk.Button(
            btn_row,
            text="✓ I Have Saved This Code",
            command=modal.destroy,
            bg=self.color_emerald,
            fg="#052e16",
            font=("Segoe UI Bold", 10),
            relief="flat",
            padx=14,
            pady=6,
            cursor="hand2"
        ).pack(side="right")

    # -------------------------------------------------------------------------
    # PC 2 REGISTRATION EXECUTION
    # -------------------------------------------------------------------------
    def _execute_pc2_register_thread(self):
        if not self.google_token:
            messagebox.showerror("Authentication Required", "Please sign in with Google first.")
            return

        dev_name = self.pc2_name_entry.get().strip() or "Secondary PC"
        self.btn_register_pc2.config(state="disabled", text="Registering Device...")

        tok = self.google_token

        def worker():
            try:
                res = self.client.register_second_pc(
                    google_id_token=tok,
                    device_name=dev_name
                )
                self.root.after(0, lambda: self._on_pc2_success(res))
            except Exception as e:
                self.root.after(0, lambda: self._on_pc2_failed(str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_pc2_success(self, res: Dict[str, Any]):
        # Clear temporary token from memory immediately
        self.google_token = None
        self.google_nonce = None
        self.google_email = None

        self.btn_register_pc2.config(state="normal", text="✓ PC Registered Successfully", bg="#065f46", fg="white")
        dev_name = res.get("device_name", "Secondary PC")
        messagebox.showinfo("PC Authorized", f"This Windows PC has been authorized as creator device:\n\n{dev_name}\n\nYou can now launch the Creator Console.")
        self.client.launch_creator_console()

    def _on_pc2_failed(self, err_msg: str):
        self.btn_register_pc2.config(state="normal", text="💻 Register & Authorize This PC")
        messagebox.showerror("Registration Failed", f"Failed to register this PC:\n\n{err_msg}")

    # -------------------------------------------------------------------------
    # DEVICE MANAGEMENT & REVOCATION
    # -------------------------------------------------------------------------
    def _refresh_device_list_async(self):
        def worker():
            try:
                devs = self.client.list_devices()
                self.root.after(0, lambda: self._populate_device_list(devs))
            except Exception as e:
                self.root.after(0, lambda: self.footer_label.config(text=f"Device list: {str(e)}", fg=self.color_amber))

        threading.Thread(target=worker, daemon=True).start()

    def _populate_device_list(self, devs: List[Dict[str, Any]]):
        self._devices_cache = devs
        self.device_listbox.delete(0, tk.END)
        for d in devs:
            name = d.get("device_name") or "Windows PC"
            stat = d.get("status", "UNKNOWN")
            mark = "✓" if stat == "AUTHORIZED" else "✗"
            # Show human-readable device name to user; internal ID kept for system use
            self.device_listbox.insert(tk.END, f"[{mark}] {name} ({stat})")

    def _revoke_selected_device(self):
        sel = self.device_listbox.curselection()
        if not sel:
            messagebox.showinfo("Select Device", "Please select a device from the list.")
            return

        idx = sel[0]
        if idx >= len(self._devices_cache):
            return

        target_device = self._devices_cache[idx]
        did = target_device.get("device_id")
        name = target_device.get("device_name") or "Selected Device"

        if messagebox.askyesno("Confirm Revocation", f"Are you sure you want to revoke '{name}'?\nThis action will immediately disable creator authority on that device."):
            try:
                res = self.client.revoke_device(did)
                messagebox.showinfo("Revoked", f"Device '{name}' has been revoked.")
                self._refresh_device_list_async()
            except Exception as e:
                messagebox.showerror("Error", str(e))

    # -------------------------------------------------------------------------
    # LAUNCH CONSOLE
    # -------------------------------------------------------------------------
    def _launch_console(self):
        ok = self.client.launch_creator_console()
        if not ok:
            messagebox.showerror("Launch Error", "Failed to launch Creator Console. Please ensure setup is complete and authority is ACTIVE.")


def launch_gui():
    """Launches the TARA Setup Graphical User Interface."""
    root = tk.Tk()
    app = SetupGUI(root)
    root.mainloop()
