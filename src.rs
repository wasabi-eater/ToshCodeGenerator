use core::ops::*;
use core::cell::{Cell, RefCell};
use std::marker::PhantomData;

#[derive(Clone)]
pub struct Expr<'a, T>{
    statements: Vec<String>,
    expr: String,
    phantom: PhantomData<&'a T>
}

impl From<f64> for Expr<'static, f64> {
    fn from(n: f64) -> Self {
        Expr{
            statements: vec![],
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
            expr: format!("(-{})", self.expr),
            phantom: PhantomData
        }
    }
}
impl<'a> Add for Expr<'a, f64> {
    type Output = Self;
    fn add(self, mut other: Self) -> Self {
        let mut statements = self.statements;
        statements.append(&mut other.statements);
        Expr {
            statements,
            expr: format!("({} + {})", self.expr, other.expr),
            phantom: PhantomData
        }
    }
}
impl<'a> Sub for Expr<'a, f64> {
    type Output = Self;
    fn sub(self, mut other: Self) -> Self {
        let mut statements = self.statements;
        statements.append(&mut other.statements);
        Expr {
            statements,
            expr: format!("({} - {})", self.expr, other.expr),
            phantom: PhantomData
        }
    }
}
impl<'a> Mul for Expr<'a, f64> {
    type Output = Self;
    fn mul(self, mut other: Self) -> Self {
        let mut statements = self.statements;
        statements.append(&mut other.statements);
        Expr {
            statements,
            expr: format!("({} * {})", self.expr, other.expr),
            phantom: PhantomData
        }
    }
}
impl<'a> Div for Expr<'a, f64> {
    type Output = Self;
    fn div(self, mut other: Self) -> Self {
        let mut statements = self.statements;
        statements.append(&mut other.statements);
        Expr {
            statements,
            expr: format!("({} / {})", self.expr, other.expr),
            phantom: PhantomData
        }
    }
}
impl<'a> From<&'a str> for Expr<'static, String> {
    fn from(s: &str) -> Self {
        Expr {
            statements: vec![],
            expr: "\"".to_string() + s + "\"",
            phantom: PhantomData
        }
    }
}

pub struct Statements{
    stack: String,
    statements: RefCell<Vec<String>>,
    var_count: Cell<usize>
}
impl Statements{
    pub fn new(stack: impl Into<String>) -> Statements {
        Statements {stack: stack.into(), statements: RefCell::new(vec![]), var_count: Cell::new(0)}
    }
    pub fn create_tosh(self) -> String{
        format!("repeat {}\ninsert \"\" at 1 of {}\nend\n", self.var_count.get(), self.stack) +
        &self.statements.into_inner().join("\n") +
        &format!("\nrepeat {}\ndelete 1 of {}\nend\n", self.var_count.get(), self.stack)
    }
    pub fn var<'a, T>(&'a self, expr: Expr<'a, T>) -> Variable<'a, T> {
        self.var_count.set(self.var_count.get() + 1);
        let number = self.var_count.get();
        self.statements.borrow_mut().push(format!("replace item {} of {} with {}", number, self.stack, expr.expr));
        Variable{statements: &self, phantom: PhantomData, number}
    }
}

pub struct Variable<'a, T>{
    statements: &'a Statements,
    phantom: PhantomData<T>,
    number: usize
}
impl<'a, T> Variable<'a, T> {
    pub fn get(&'a self) -> Expr<'a, T>{
        Expr{
            statements: vec![],
            expr: format!("(item {} of {})", self.number, self.statements.stack),
            phantom: PhantomData
        }
    }
    pub fn set(&'a self, mut expr: Expr<'a, T>) {
        let mut statements = self.statements.statements.borrow_mut();
        statements.append(&mut expr.statements);
        statements.push(format!("replace item {} of {} with {}", self.number, self.statements.stack, expr.expr));
    }
}

fn main(){
    let statements = Statements::new("stack");
    let v1 = statements.var(Expr::from(1.5));
    v1.set(v1.get() + Expr::from(2.1));
    let v2 = statements.var(Expr::from("abc"));
    v2.set(Expr::from("def"));
    print!("{}", statements.create_tosh());
}
