use std::rc::Rc;
use core::ops::*;
use core::cell::Cell;
use std::marker::PhantomData;
enum ExprTree{
    Num(String),
    Str(String),
    BinOp{
        left: Box<ExprTree>,
        right: Box<ExprTree>,
        op: &'static str
    },
    DataItemOf{
        index: Box<ExprTree>,
        list: Rc<str>
    },
    DataAdd{
        item: Box<ExprTree>,
        list: Rc<str>
    },
    DataInsert{
        item: Box<ExprTree>,
        index: Box<ExprTree>,
        list: Rc<str>
    },
    DataDelete{
        index: Box<ExprTree>,
        list: Rc<str>
    },
    DataLengthOf{
        list: Rc<str>
    }
}
impl ExprTree{
    pub fn as_tosh_code(self) -> String{
        match self {
            ExprTree::Num(n) => n,
            ExprTree::Str(s) => format!("\"{}\"", s),
            ExprTree::BinOp{left, op, right} => format!("({} {} {})", left.as_tosh_code(), op, right.as_tosh_code()),
            ExprTree::DataItemOf{index, list} => format!("(item {} of {})", index.as_tosh_code(), list),
            ExprTree::DataAdd{item, list} => format!("add {} to {}", item.as_tosh_code(), list),
            ExprTree::DataInsert{item, index, list} => format!("insert {} at {} of {}", item.as_tosh_code(), index.as_tosh_code(), list),
            ExprTree::DataDelete{index, list} => format!("delete {} of {}", index.as_tosh_code(), list),
            ExprTree::DataLengthOf{list} => format!("(length of {})", list)
        }
    }
}
pub trait FieldSized {
    fn size() -> usize;
}
impl FieldSized for () {
    fn size() -> usize {
        0
    }
}
impl FieldSized for f64 {
    fn size() -> usize {
        1
    }
}
impl FieldSized for String {
    fn size() -> usize {
        1
    }
}
impl<T1: FieldSized, T2: FieldSized> FieldSized for (T1, T2) {
    fn size() -> usize {
        T1::size() + T2::size()
    }
}
struct StackDelete{
    stack: Rc<str>
}
pub struct Expr<'a, T : FieldSized>{
    statements: Vec<ExprTree>,
    post_process: Vec<StackDelete>,
    expr: Vec<ExprTree>,
    phantom: PhantomData<&'a T>
}
impl<'a, T : FieldSized> Expr<'a, T>{
    fn create_tosh(self) -> String {
        format!("{}\n{}", self.statements.into_iter().map(|s| s.as_tosh_code()).collect::<Vec<_>>().join("\n"),
            self.post_process.into_iter().map(|s| ExprTree::DataDelete{
                index: ExprTree::Num("1".into()).into(),
                list: s.stack
            }.as_tosh_code()).collect::<Vec<_>>().join("\n"))
    }
    pub fn var<'b, 'c, T2 : FieldSized>(mut self, stack: &'c Stack, f: impl FnOnce(Variable<'c, T>) -> Expr<'b, T2>) -> Expr<'b, T2>
        where T: 'c {
        let expr_count = self.expr.len();
        let number = (stack.var_count.get() + 1 .. stack.var_count.get() + expr_count + 1).collect();
        stack.var_count.set(stack.var_count.get() + expr_count);
        let mut r = f(Variable{stack: &stack, phantom: PhantomData, number});
        stack.var_count.set(stack.var_count.get() - 1);
        let mut statements = self.statements;
        if self.expr.len() == 1 {
            statements.push(ExprTree::DataInsert{
                item: self.expr.pop().unwrap().into(),
                index: ExprTree::Num("0".into()).into(),
                list: stack.name.clone()});
        }
        else {
            for expr in self.expr.into_iter().rev() {
                statements.push(ExprTree::DataAdd{item: expr.into(), list: stack.name.clone()});
            }
            for _ in 0..expr_count {
                statements.push(ExprTree::DataInsert{
                    item: ExprTree::DataItemOf{
                        index: ExprTree::DataLengthOf{list: stack.name.clone()}.into(),
                        list: stack.name.clone(),
                    }.into(),
                    index: ExprTree::Num("1".into()).into(),
                    list: stack.name.clone()
                });
                statements.push(ExprTree::DataDelete{
                    index: ExprTree::DataLengthOf{list: stack.name.clone()}.into(),
                    list: stack.name.clone()
                });
            }
        }
        for StackDelete{stack: _stack} in self.post_process {
            if _stack == stack.name {
                statements.push(ExprTree::DataDelete{
                    index: ExprTree::Num((expr_count + 1).to_string()).into(),
                    list: _stack
                });
            }
            else {
                statements.push(ExprTree::DataDelete{
                    index: ExprTree::Num("1".into()).into(),
                    list: _stack
                });
            }
        }
        statements.append(&mut r.statements);
        r.post_process.push(StackDelete{stack: stack.name.clone()});
        Expr {
            statements,
            post_process: r.post_process,
            expr: r.expr,
            phantom: PhantomData
        }
    }
}
impl<'a> From<()> for Expr<'a, ()> {
    fn from(_: ()) -> Self {
        Expr {
            statements: vec![],
            post_process: vec![],
            expr: vec![],
            phantom: PhantomData
        }
    }
}
impl <'a, T1: Into<Expr<'a, A>>, T2: Into<Expr<'a, B>>, A: FieldSized, B: FieldSized> From<(T1, T2)> for Expr<'a, (A, B)> {
    fn from(tuple: (T1, T2)) -> Self {
        let (mut x, mut y) = (tuple.0.into(), tuple.1.into());
        let mut statements = x.statements;
        statements.append(&mut y.statements);
        let mut post_process = y.post_process;
        post_process.append(&mut x.post_process);
        let mut expr = x.expr;
        expr.append(&mut y.expr);
        Expr {
            statements,
            post_process,
            expr,
            phantom: PhantomData
        }
    }
}
impl <'a, T1: FieldSized, T2: FieldSized> Expr<'a, (T1, T2)> {
    pub fn item0(mut self) -> Expr<'a, T1> {
        let _ = self.expr.split_off(T1::size());
        Expr{
            statements: self.statements,
            expr: self.expr,
            post_process: self.post_process,
            phantom: PhantomData
        }
    }
    pub fn item1(mut self) -> Expr<'a, T2> {
        let t2_expr = self.expr.split_off(T1::size());
        Expr{
            statements: self.statements,
            expr: t2_expr,
            post_process: self.post_process,
            phantom: PhantomData
        }
    }
}
impl<'a> From<f64> for Expr<'a, f64> {
    fn from(n: f64) -> Self {
        Expr{
            statements: vec![],
            post_process: vec![],
            expr: vec![ExprTree::Num(n.to_string())],
            phantom: PhantomData
        }
    }
}
impl<'a> Neg for Expr<'a, f64>{
    type Output = Self;
    fn neg(mut self) -> Self {
        Expr{
            statements: self.statements,
            post_process: self.post_process,
            expr: vec![ExprTree::BinOp{
                left: ExprTree::Num("0".into()).into(),
                op: "-",
                right: self.expr.pop().unwrap().into()
            }],
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
            expr: vec![ExprTree::BinOp{
                left: self.expr.pop().unwrap().into(),
                op: "+",
                right: other.expr.pop().unwrap().into()
            }],
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
            expr: vec![ExprTree::BinOp{
                left: self.expr.pop().unwrap().into(),
                op: "-",
                right: other.expr.pop().unwrap().into()
            }],
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
            expr: vec![ExprTree::BinOp{
                left: self.expr.pop().unwrap().into(),
                op: "*",
                right: other.expr.pop().unwrap().into()
            }],
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
            expr: vec![ExprTree::BinOp{
                left: self.expr.pop().unwrap().into(),
                op: "/",
                right: other.expr.pop().unwrap().into()
            }],
            phantom: PhantomData
        }
    }
}
impl<'a, 'b> From<&'a str> for Expr<'b, String> {
    fn from(s: &str) -> Self {
        Expr {
            statements: vec![],
            post_process: vec![],
            expr: vec![ExprTree::Str(s.into())],
            phantom: PhantomData
        }
    }
}

pub struct Stack{
    name: Rc<str>,
    var_count: Cell<usize>
}
impl Stack{
    pub fn new(name: &str) -> Self {
        Self {name: name.into(), var_count: Cell::new(0)}
    }
}

pub struct Variable<'a, T : FieldSized>{
    stack: &'a Stack,
    phantom: PhantomData<&'a T>,
    number: Vec<usize>
}
impl<'a, T : FieldSized> Variable<'a, T> {
    pub fn get(&self) -> Expr<'a, T>{
        Expr{
            statements: vec![],
            post_process: vec![],
            expr: self.number.iter()
                .map(|number| ExprTree::DataItemOf{
                    index: ExprTree::Num((self.stack.var_count.get() - number + 1).to_string()).into(),
                    list: self.stack.name.clone()
                })
                .collect(),
            phantom: PhantomData
        }
    }
}

macro_rules! action {
    {$stack:expr; let! $v:pat = $e:expr; $(let! $v2:pat = $e2:expr;)* $e3:expr} => {
        $e.var(&$stack, |$v| action!{$stack; $(let! $v2 = $e2;)* $e3})
    };
    {$stack:expr; $e:expr} => ($e);
}

fn main(){
    let stack = Stack::new("stack");
    let expr = action!{
        stack;
        let! z = action!{
            stack;
            let! x = Expr::from(0.4) + Expr::from(3.2);
            let! y = x.get() + Expr::from(3.0);
            x.get() * y.get()
        };
        let! w = Expr::from(((z.get(), "Hello World"), 3.2));
        let! w2 = w.get().item0();
        Expr::from(())
    };
    println!("{}", expr.create_tosh());
}
