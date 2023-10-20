use std::cell::Cell;

pub struct Var<'a>{
    pub value: f64,
    pub grad: Cell<f64>,
    visited: Cell<bool>,
    pub ch1: Option<&'a Var<'a>>,
    pub ch2: Option<&'a Var<'a>>,
    operation: Box<dyn Operation + 'a>,
}

trait Operation{
    fn op(&self, a: f64, b: f64) -> f64;
    fn grad(&self, a: f64, b: f64) -> f64;
}

struct NoOP;
struct AddOP;
struct NegOp;
struct MulOp;
struct PowOp{
    p: f64,
}

impl Operation for NoOP{
    fn op(&self, _: f64, _: f64) -> f64 { return 0.0 }
    fn grad(&self, _: f64, _: f64) -> f64 { return 0.0 }
}

impl Operation for AddOP{
    fn op(&self, a: f64, b: f64) -> f64 { return a + b }
    fn grad(&self, _: f64, _: f64) -> f64 { return 1.0; }
}

impl Operation for NegOp{
    fn op(&self, a: f64, _: f64) -> f64 { return -a; }
    fn grad(&self, _: f64, _: f64) -> f64 { return -1.0; }
}

impl Operation for MulOp{
    fn op(&self, a: f64, b: f64) -> f64 { return a * b; }
    fn grad(&self, _: f64, b: f64) -> f64 { return b; }
}

impl Operation for PowOp{
    fn op(&self, a: f64, _: f64) -> f64 { return a.powf(self.p); }
    fn grad(&self, a: f64, _: f64) -> f64 { return self.p * a.powf(self.p - 1.0); }
}

pub fn new_var<'a>(value: f64) -> Var<'a>{
    return Var{
        value,
        grad: Cell::new(0.0),
        visited: Cell::new(false),
        ch1: None,
        ch2: None,
        operation: Box::new(NoOP)
    }
}

fn combine<'a>(x: &'a Var<'a>, y: &'a Var<'a>, op: impl Operation + 'a) -> Var<'a>{
    return Var{
        value: op.op(x.value, y.value),
        grad: Cell::new(0.0),
        visited: Cell::new(false),
        ch1: Some(x),
        ch2: Some(y),
        operation: Box::new(op),
    }
}

impl<'a> Var<'a> {
    pub fn add(&'a self, o: &'a Var<'a>) -> Var<'a> {
        return combine(self, o, AddOP{});
    }

    pub fn neg(&'a self) -> Var<'a> {
        return Var{
            value: NegOp{}.op(self.value, 0.0),
            grad: Cell::new(0.0),
            visited: Cell::new(false),
            ch1: Some(self),
            ch2: None,
            operation: Box::new(NegOp{}),
        }
    }

    pub fn mul(&'a self, o: &'a Var<'a>) -> Var<'a> {
        return combine(self, o, MulOp{})
    }

    pub fn pow(&'a self, p: f64) -> Var<'a>{
        let pow_op = PowOp{ p };
        return Var{
            value: pow_op.op(self.value, p),
            grad: Cell::new(0.0),
            visited: Cell::new(false),
            ch1: Some(self),
            ch2: None,
            operation: Box::new(pow_op),
        }
    }

    fn reset_vis(&self){
        self.visited.set(false);
        if self.ch1.is_some() { self.ch1.unwrap().reset_vis(); }
        if self.ch2.is_some() { self.ch2.unwrap().reset_vis(); }
    }

    pub fn backward(&self){
        self.grad.set(1.0);
        self.reset_vis();
        self._backward();
    }

    fn _backward(&self){
        if self.ch1.is_some(){
            self.ch1.unwrap().grad.set(
                self.grad.get() * self.operation.grad(self.ch1.unwrap().value, self.ch2.unwrap().value) + self.ch1.unwrap().grad.get()
            );
        }
        if self.ch2.is_some(){
            self.ch2.unwrap().grad.set(
                self.grad.get() * self.operation.grad(self.ch2.unwrap().value, self.ch1.unwrap().value) + self.ch2.unwrap().grad.get()
            );
        }
        if self.ch1.is_some() { self.ch1.unwrap()._backward(); }
        if self.ch2.is_some() { self.ch2.unwrap()._backward(); }
    }
}

#[test]
fn test_all() {
    // not full
    let a = new_var(4.0);
    let x = new_var(3.0);
    let b = new_var(10.0);
    let r = a.mul(&x);
    let t = r.add(&b);
    t.backward();
    assert_eq!(x.grad.get(), a.value);
}

