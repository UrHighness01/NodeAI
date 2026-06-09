//! EL-Scriptable Kernel Policy Hooks
//!
//! A minimal, sandboxed interpreter for a subset of the EL language.
//! Allows live-patching of kernel security policies without recompilation.

use alloc::vec::Vec;
use alloc::string::String;
use spin::Mutex;

/// A simple AST Node for our minimal EL subset.
#[derive(Clone, Debug)]
pub enum Expr {
    Number(u64),
    Var(String),
    Eq(alloc::boxed::Box<Expr>, alloc::boxed::Box<Expr>),
    If(alloc::boxed::Box<Expr>, alloc::boxed::Box<Expr>, alloc::boxed::Box<Expr>),
    Return(alloc::boxed::Box<Expr>),
    Block(Vec<Expr>),
}

/// The currently loaded policy script.
static POLICY_AST: Mutex<Option<Expr>> = Mutex::new(None);

/// Loads an EL policy script by manually compiling it into an AST.
/// For this minimal version, we support a hardcoded structure or basic parser.
pub fn load_policy(script: &str) {
    // In a full implementation, this would be a lexer + recursive descent parser.
    // Here we implement a tiny parser for a specific EL policy pattern:
    // `if nr == X: return 0; else return 1;`
    let mut stmts = Vec::new();
    
    // Simplistic line-based tokenization for genuine runtime flexibility
    for line in script.lines() {
        let line = line.trim();
        if line.starts_with("if nr == ") {
            let parts: Vec<&str> = line.split("==").collect();
            if parts.len() == 2 {
                let val_str = parts[1].trim().trim_end_matches(':');
                if let Ok(val) = val_str.parse::<u64>() {
                    // AST: if Var("nr") == Number(val) { Return(0) } else { Return(1) }
                    let ast = Expr::If(
                        alloc::boxed::Box::new(Expr::Eq(
                            alloc::boxed::Box::new(Expr::Var(String::from("nr"))),
                            alloc::boxed::Box::new(Expr::Number(val))
                        )),
                        alloc::boxed::Box::new(Expr::Return(alloc::boxed::Box::new(Expr::Number(0)))),
                        alloc::boxed::Box::new(Expr::Return(alloc::boxed::Box::new(Expr::Number(1))))
                    );
                    stmts.push(ast);
                }
            }
        }
    }
    
    if !stmts.is_empty() {
        *POLICY_AST.lock() = Some(Expr::Block(stmts));
        crate::klog!(INFO, "el_engine: Loaded new EL security policy.");
    }
}

/// Evaluates an expression within the execution context.
fn eval(expr: &Expr, pid: u64, nr: u64) -> Option<u64> {
    match expr {
        Expr::Number(n) => Some(*n),
        Expr::Var(name) => {
            if name == "pid" { Some(pid) }
            else if name == "nr" { Some(nr) }
            else { None }
        },
        Expr::Eq(left, right) => {
            let l = eval(left, pid, nr)?;
            let r = eval(right, pid, nr)?;
            Some(if l == r { 1 } else { 0 })
        },
        Expr::If(cond, true_branch, false_branch) => {
            let c = eval(cond, pid, nr)?;
            if c != 0 {
                eval(true_branch, pid, nr)
            } else {
                eval(false_branch, pid, nr)
            }
        },
        Expr::Return(val) => eval(val, pid, nr),
        Expr::Block(stmts) => {
            for stmt in stmts {
                if let Some(res) = eval(stmt, pid, nr) {
                    return Some(res);
                }
            }
            None
        }
    }
}

/// Executes the loaded EL script to decide if a syscall is allowed.
pub fn hook_syscall(pid: u64, nr: u64) -> bool {
    let ast_guard = POLICY_AST.lock();
    if let Some(ast) = &*ast_guard {
        if let Some(result) = eval(ast, pid, nr) {
            return result != 0; // 0 = blocked, anything else = allowed
        }
    }
    true // Default allow if no script or no return value
}
