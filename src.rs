use core::ops::*;
use core::cell::{Cell, RefCell};
use std::marker::PhantomData;


pub struct Expr<'a, T>{
    statements: Vec<String>,
    post_process: Vec<String>,
    expr: String,
    phantom: PhantomData<&'a T>
}
impl<'a, T> Expr<'a, T>{
    fn create_tosh(self) -> String {
        format!("{}\n{}", self.statements.join("\n"), self.post_process.join("\n"))
    }
}
impl From<f64> for Expr<'static, f64> {
    fn from(n: f64) -> Self {
        Expr{
            statements: vec![],
            post_process: vec![],
            expr: n.to_string(),
            phantom: PhantomData
        }
    }
}
impl<'a> Neg for Expr<'a, f64>{
    type Output = Self;
    fn neg(self) -> Self {
        Expr{
            statements: self.statements,
            post_process: self.post_process,
            expr: format!("(-{})", self.expr),
            phantom: PhantomData
        }
    }
}
impl<'a> Add for Expr<'a, f64> {
    type Output = Self;
    fn add(mut self, mut other: Self) -> Self {
        let mut statements = self.statements;
        statements.append(&mut other.statements);
        let mut post_process = other.post_process;
        post_process.append(&mut self.post_process);
        Expr {
            statements,
            post_process,
            expr: format!("({} + {})", self.expr, other.expr),
            phantom: PhantomData
        }
    }
}
impl<'a> Sub for Expr<'a, f64> {
    type Output = Self;
    fn sub(mut self, mut other: Self) -> Self {
        let mut statements = self.statements;
        statements.append(&mut other.statements);
        let mut post_process = other.post_process;
        post_process.append(&mut self.post_process);
        Expr {
            statements,
            post_process,
            expr: format!("({} - {})", self.expr, other.expr),
            phantom: PhantomData
        }
    }
}
impl<'a> Mul for Expr<'a, f64> {
    type Output = Self;
    fn mul(mut self, mut other: Self) -> Self {
        let mut statements = self.statements;
        statements.append(&mut other.statements);
        let mut post_process = other.post_process;
        post_process.append(&mut self.post_process);
        Expr {
            statements,
            post_process,
            expr: format!("({} * {})", self.expr, other.expr),
            phantom: PhantomData
        }
    }
}
impl<'a> Div for Expr<'a, f64> {
    type Output = Self;
    fn div(mut self, mut other: Self) -> Self {
        let mut statements = self.statements;
        statements.append(&mut other.statements);
        let mut post_process = other.post_process;
        post_process.append(&mut self.post_process);
        Expr {
            statements,
            post_process,
            expr: format!("({} / {})", self.expr, other.expr),
            phantom: PhantomData
        }
    }
}
impl<'a> From<&'a str> for Expr<'static, String> {
    fn from(s: &str) -> Self {
        Expr {
            statements: vec![],
            post_process: vec![],
            expr: "\"".to_string() + s + "\"",
            phantom: PhantomData
        }
    }
}

pub struct Stack{
    stack: String,
    var_count: Cell<usize>
}
impl Stack{
    pub fn new(stack: impl Into<String>) -> Self {
        Self {stack: stack.into(), var_count: Cell::new(0)}
    }
    pub fn var<'a, T>(&'a self, expr: Expr<'a, T>, f: impl for<'b> FnOnce(Variable<'b, T>) -> Expr<'b, T>) -> Expr<'a, T> {
        self.var_count.set(self.var_count.get() + 1);
        let number = self.var_count.get();
        let mut r = f(Variable{stack: &self, phantom: PhantomData, number});
        self.var_count.set(self.var_count.get() - 1);
        r.statements.insert(0, format!("insert {} at 0 of {}", expr.expr, self.stack));
        r.post_process.push(format!("delete at 0 of {}", self.stack));
        r
    }
}

pub struct Variable<'a, T>{
    stack: &'a Stack,
    phantom: PhantomData<&'a T>,
    number: usize
}
impl<'a, T> Variable<'a, T> {
    pub fn get(&self) -> Expr<'a, T>{
        Expr{
            statements: vec![],
            post_process: vec![],
            expr: format!("(item {} of {})", self.stack.var_count.get() - self.number, self.stack.stack),
            phantom: PhantomData
        }
    }
}

fn main(){
    let stack = Stack::new("stack");
    let expr = stack.var(Expr::from(0.4) + Expr::from(3.2), |v| v.get() * Expr::from(2.3));
    print!("{}", expr.create_tosh());
}
